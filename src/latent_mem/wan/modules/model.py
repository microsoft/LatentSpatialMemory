# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import repeat

from .attention import flash_attention

__all__ = ['WanModel']


def modulate(x, shift, scale):
    """
    Apply affine transformation for AdaLN.
    Args:
        x: input tensor [B, L, C]
        shift: shift parameter [B, L, C]
        scale: scale parameter [B, L, C]
    Returns:
        Modulated tensor [B, L, C]
    """
    return x * (1 + scale) + shift

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# @amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanGanCrossAttention(WanSelfAttention):

    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        qq = self.norm_q(self.q(context)).view(b, 1, -1, d)

        kk = self.norm_k(self.k(x)).view(b, -1, n, d)
        vv = self.v(x).view(b, -1, n, d)

        # compute attention
        x = flash_attention(qq, kk, vv)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(
            dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 vace_block_id=None,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.vace_block_id = vace_block_id

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)

        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        vace_hints=None,
        vace_context_scale=1.0,
        vace_hint_offset=0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            vace_hints(List[Tensor], optional): VACE hint features for injection
            vace_context_scale(float): Scale factor for VACE hint features
            vace_hint_offset(int): Token offset for VACE hint injection (for Spatia: skip R reference tokens)
        """
        # Handle per-frame timestep [B, F, 6, C] vs uniform [B, 6, C]
        if e.dim() == 4:
            # Per-frame timestep: e is [B, F, 6, dim]
            num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
            e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)  # 6 x [B, F, 1, dim]

            # Self-attention: unflatten x to [B, F, frame_seqlen, dim], apply modulation, flatten back
            x_norm = self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            y = self.self_attn((x_norm * (1 + e[1]) + e[0]).flatten(1, 2), seq_lens, grid_sizes, freqs)
            # Unflatten y, multiply by e[2], then flatten before adding to x
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

            # Cross-attention & FFN
            x = x + self.cross_attn(x, context, context_lens)
            x_norm2 = self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            y = self.ffn((x_norm2 * (1 + e[4]) + e[3]).flatten(1, 2))
            # Unflatten y, multiply by e[5], then flatten before adding to x
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        else:
            # Uniform timestep: e is [B, 6, dim]
            e = (self.modulation + e).chunk(6, dim=1)

            # Self-attention
            y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs)
            x = x + y * e[2]

            # Cross-attention & FFN
            x = x + self.cross_attn(x, context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            x = x + y * e[5]

        # VACE hint injection
        if self.vace_block_id is not None and vace_hints is not None:
            hint = vace_hints[self.vace_block_id] * vace_context_scale
            if vace_hint_offset > 0:
                # Spatia: Add hint to positions [offset:offset+hint_len] (skip reference tokens)
                # hint has P+T tokens, x has R+P+T tokens
                hint_len = hint.shape[1]
                x[:, vace_hint_offset:vace_hint_offset + hint_len, :] = (
                    x[:, vace_hint_offset:vace_hint_offset + hint_len, :] + hint
                )
            else:
                # Original VACE: hint and x have same size
                x = x + hint

        return x


class VaceControlBlock(WanAttentionBlock):
    """
    VACE Control Block that processes control signals.
    Extends WanAttentionBlock with additional projection layers for feature stacking.
    """
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            vace_ctrl_block_id=0,
            **kwargs
    ):
        # Don't pass vace_block_id to parent (control blocks don't inject hints)
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps, vace_block_id=None, **kwargs)
        self.vace_ctrl_block_id = vace_ctrl_block_id

        # Zero-initialized projection layers for VACE feature stacking
        if vace_ctrl_block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        """
        Forward pass for VACE control block.

        Args:
            c: Control features (stacked tensor or single tensor)
            x: Input features from main model (for block 0 initialization)
            **kwargs: Additional arguments for parent forward

        Returns:
            Updated control features stack
        """
        if self.vace_ctrl_block_id == 0:
            # First block: initialize control features from main features
            c = self.before_proj(c) + x
            all_c = []
        else:
            # Subsequent blocks: unstack and process last feature
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        # Process through attention block
        c = super().forward(c, **kwargs)

        # Create skip connection and stack
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C] (uniform) or [B, F, C] (per-frame)
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        if e.dim() == 3:
            # Per-frame timestep: e is [B, F, dim]
            num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
            e = (self.modulation.unsqueeze(1) + e.unsqueeze(2)).chunk(2, dim=2)  # 2 x [B, F, 1, dim]
            x_norm = self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            x = self.head((x_norm * (1 + e[1]) + e[0]).flatten(1, 2))
        else:
            # Uniform timestep: e is [B, dim]
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class RegisterTokens(nn.Module):
    def __init__(self, num_registers: int, dim: int):
        super().__init__()
        self.register_tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.rms_norm = WanRMSNorm(dim, eps=1e-6)

    def forward(self):
        return self.rms_norm(self.register_tokens)

    def reset_parameters(self):
        nn.init.normal_(self.register_tokens, std=0.02)


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 vace_layers=None,
                 vace_in_dim=None,
                 **kwargs):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            vace_layers (`list`, *optional*, defaults to None):
                Layer indices where VACE hints are injected. If None, VACE is disabled.
                Example: [0, 2, 4, 6, ...] for every other layer
            vace_in_dim (`int`, *optional*, defaults to None):
                Input channels for VACE control signals. If None, uses in_dim.
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        # VACE configuration
        self.vace_layers = vace_layers
        self.vace_in_dim = in_dim if vace_in_dim is None else vace_in_dim
        self.enable_vace = vace_layers is not None and len(vace_layers) > 0

        if self.enable_vace:
            assert 0 in vace_layers, "VACE layer 0 must be included for proper initialization"
            self.vace_layers_mapping = {i: n for n, i in enumerate(vace_layers)}

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.local_attn_size = 21

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        # ti2v uses t2v_cross_attn (Wan2.2-5B structure, no separate k_img/v_img layers)
        if model_type in ['t2v', 'ti2v']:
            cross_attn_type = 't2v_cross_attn'
        else:  # i2v
            cross_attn_type = 'i2v_cross_attn'

        # Create blocks with optional VACE block IDs
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                window_size, qk_norm, cross_attn_norm, eps,
                vace_block_id=self.vace_layers_mapping[i] if self.enable_vace and i in self.vace_layers else None,
                **kwargs
            )
            for i in range(num_layers)
        ])

        # VACE control blocks
        if self.enable_vace:
            self.vace_blocks = nn.ModuleList([
                VaceControlBlock(
                    't2v_cross_attn', dim, ffn_dim, num_heads, window_size,
                    qk_norm, cross_attn_norm, eps,
                    vace_ctrl_block_id=i, **kwargs
                )
                for i in range(len(self.vace_layers))
            ])

            # VACE patch embedding for control signals
            self.vace_patch_embedding = nn.Conv3d(
                self.vace_in_dim, dim, kernel_size=patch_size, stride=patch_size
            )
        else:
            self.vace_blocks = None
            self.vace_patch_embedding = None

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        # #! dzc
        # self.init_weights()

        self.gradient_checkpointing = False

    # 在 CausalWanModel 内部添加或替换该方法
    def _set_gradient_checkpointing(self, *args, **kwargs):
        """
        兼容 Diffusers 新旧两个调用方式：
        - 旧: _set_gradient_checkpointing(value: bool)
        - 新: _set_gradient_checkpointing(enable: bool = True, gradient_checkpointing_func: Optional[Callable] = None)
        """
        # 1) 解析参数，兼容两种风格
        if "enable" in kwargs or "gradient_checkpointing_func" in kwargs:
            enable = kwargs.get("enable", True)
            grad_ckpt_func = kwargs.get("gradient_checkpointing_func", None)
        elif len(args) >= 1 and isinstance(args[0], bool):
            enable = args[0]
            grad_ckpt_func = None
        else:
            enable = True
            grad_ckpt_func = None

        # 2) 递归地把设置应用到子模块（如果它们支持）
        def _apply(module):
            # 一些自定义模块用这个 flag
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enable

            # Diffusers/Transformers 常见写法：提供 set_gradient_checkpointing 方法
            if enable and hasattr(module, "set_gradient_checkpointing"):
                # 新接口希望传函数；旧接口通常不需要
                try:
                    module.set_gradient_checkpointing(grad_ckpt_func)  # 新式
                except TypeError:
                    module.set_gradient_checkpointing(enable=True)     # 旧式兜底

        # 把设置应用到整棵模型
        for m in self.modules():
            _apply(m)

    def forward(
        self,
        *args,
        **kwargs
    ):
        # if kwargs.get('classify_mode', False) is True:
        # kwargs.pop('classify_mode')
        # return self._forward_classify(*args, **kwargs)
        # else:
        return self._forward(*args, **kwargs)

    def _forward(
        self,
        x,
        t,
        context,
        seq_len,
        classify_mode=False,
        concat_time_embeddings=False,
        register_tokens=None,
        cls_pred_branch=None,
        gan_ca_blocks=None,
        clip_fea=None,
        y=None,
        vace_context=None,
        vace_context_scale=1.0,
        vace_hint_offset=0,
        **kwargs,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            vace_context (List[Tensor], *optional*):
                VACE control signals, list of [C_vace, F_vace, H, W] where F_vace may be < F_main
            vace_context_scale (float):
                Scale factor for VACE hint injection
            vace_hint_offset (int):
                Token offset for VACE hints (for Spatia: skip R*H*W reference tokens)

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # ti2v uses y (first frame conditioning) but not clip_fea
        if self.model_type == 'ti2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
        ])

        # Wan2.2-5B
        if self.model_type == "ti2v":
            # time embeddings
            if t.dim() == 1:
                t = t.expand(t.size(0), seq_len)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                bt = t.size(0)
                t = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        else:
            # time embeddings
            # Support per-frame timestep [B, F] like causal model
            if t.dim() == 2:
                # Per-frame timestep: t is [B, F]
                t_shape = t.shape
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
                e0 = self.time_projection(e).unflatten(
                    1, (6, self.dim)).unflatten(dim=0, sizes=t_shape)  # [B, F, 6, dim]
                # Reshape e to [B, F, dim] for head
                e = e.unflatten(dim=0, sizes=t_shape)  # [B, F, dim]
            else:
                # Single timestep: t is [B]
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # [B, 6, dim]


        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)


        # arguments
        base_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens
            )

        # VACE hint generation
        vace_hints = None
        if self.enable_vace and vace_context is not None:
            # Embed VACE context: each u is [C, P+T, H, W] for Spatia
            c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
            c = [u.flatten(2).transpose(1, 2) for u in c]  # list of [1, (P+T)*h*w, dim]
            c = torch.cat(c)  # [B, vace_seq_len, dim] where vace_seq_len = (P+T) * tokens_per_frame

            # For Spatia: VACE processes P+T frames while main model has R+P+T frames
            # We need to slice x to only P+T portion for VACE block initialization
            vace_seq_len = c.shape[1]  # (P+T) * tokens_per_frame
            if vace_seq_len < seq_len:
                # Spatia case: main model has more frames than VACE
                # Use vace_hint_offset to get the P+T portion of x
                x_for_vace = x[:, vace_hint_offset:vace_hint_offset + vace_seq_len, :]
            else:
                # Original VACE case: VACE and main model have same frames
                x_for_vace = x

            # Prepare kwargs for VACE blocks
            # For per-frame timestep, also slice e0 to P+T frames
            vace_e0 = e0
            if e0.dim() == 4 and vace_seq_len < seq_len:
                # Per-frame timestep with Spatia: e0 is [B, R+P+T, 6, dim]
                # Need to slice to [B, P+T, 6, dim] for VACE
                total_frames = e0.shape[1]
                tokens_per_frame = seq_len // total_frames
                num_vace_frames = vace_seq_len // tokens_per_frame
                num_ref_frames = total_frames - num_vace_frames
                vace_e0 = e0[:, num_ref_frames:]  # [B, P+T, 6, dim]

            vace_kwargs = dict(x=x_for_vace, e=vace_e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                               freqs=self.freqs, context=context, context_lens=context_lens)
            # Process through VACE control blocks
            for block in self.vace_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    c = torch.utils.checkpoint.checkpoint(
                        lambda c_in, **kw: block(c_in, **kw),
                        c, **vace_kwargs,
                        use_reentrant=False,
                    )
                else:
                    c = block(c, **vace_kwargs)

            # Extract hints (all except the last stacked feature)
            vace_hints = torch.unbind(c)[:-1]

        # Add VACE hints to forward kwargs
        forward_kwargs = base_kwargs.copy()
        forward_kwargs['vace_hints'] = vace_hints
        forward_kwargs['vace_context_scale'] = vace_context_scale
        forward_kwargs['vace_hint_offset'] = vace_hint_offset

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        final_x = None
        if classify_mode:
            assert register_tokens is not None
            assert gan_ca_blocks is not None
            assert cls_pred_branch is not None

            final_x = []
            registers = repeat(register_tokens(), "n d -> b n d", b=x.shape[0])
            # x = torch.cat([registers, x], dim=1)

        gan_idx = 0
        for ii, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **forward_kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **forward_kwargs)

            if classify_mode and ii in [13, 21, 29]:
                gan_token = registers[:, gan_idx: gan_idx + 1]
                final_x.append(gan_ca_blocks[gan_idx](x, gan_token))
                gan_idx += 1

        if classify_mode:
            final_x = torch.cat(final_x, dim=1)
            if concat_time_embeddings:
                final_x = cls_pred_branch(torch.cat([final_x, 10 * e[:, None, :]], dim=1).view(final_x.shape[0], -1))
            else:
                final_x = cls_pred_branch(final_x.view(final_x.shape[0], -1))

        # head
        x = self.head(x, e) # [1, 32760, 1536], [1, 1536]

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        if classify_mode:
            return torch.stack(x), final_x

        return torch.stack(x)



    def unpatchify(self, x, grid_sizes, c=None):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim if c is None else c
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
