# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config

from .model import WanAttentionBlock, WanModel, sinusoidal_embedding_1d

__all__ = ["VaceWanModel", "VaceWanAttentionBlock", "BaseWanAttentionBlock"]


class VaceWanAttentionBlock(WanAttentionBlock):
    """
    VACE Attention Block that processes control signals.
    Extends WanAttentionBlock with additional projection layers for feature injection.
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
        block_id=0,
        **kwargs,
    ):
        super().__init__(
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size,
            qk_norm,
            cross_attn_norm,
            eps,
            **kwargs,
        )
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        """
        Forward pass for VACE block.

        Args:
            c: Control features (stacked tensor or single tensor)
            x: Input features from main model
            **kwargs: Additional arguments for parent forward

        Returns:
            Updated control features stack
        """
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class BaseWanAttentionBlock(WanAttentionBlock):
    """
    Base attention block with VACE hint injection support.
    Extends WanAttentionBlock to accept and integrate hint features.
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
        block_id=None,
        **kwargs,
    ):
        super().__init__(
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size,
            qk_norm,
            cross_attn_norm,
            eps,
            **kwargs,
        )
        self.block_id = block_id

    def forward(self, x, hints=None, context_scale=1.0, **kwargs):
        """
        Forward pass with optional hint injection.

        Args:
            x: Input features [B, L, C]
            hints: Optional list of hint features for injection
            context_scale: Scale factor for hint features
            **kwargs: Additional arguments for parent forward

        Returns:
            Updated features [B, L, C]
        """
        x = super().forward(x, **kwargs)
        if self.block_id is not None and hints is not None:
            x = x + hints[self.block_id] * context_scale
        return x


class VaceWanModel(WanModel):
    """
    VACE-enhanced Wan Model for controllable video generation.

    This model extends WanModel with VACE (Video-Aware Control Enhancement) support,
    allowing additional control signals to guide the generation process.

    Args:
        vace_layers (list, optional): Layer indices where VACE hints are injected.
            Defaults to every other layer [0, 2, 4, ...].
        vace_in_dim (int, optional): Input channels for VACE control signals.
            Defaults to in_dim.
        attach_vace (bool, optional): Whether to attach VACE control branch.
            Defaults to True.
        All other args: Same as WanModel parent class.
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["BaseWanAttentionBlock", "VaceWanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        vace_layers=None,
        vace_in_dim=None,
        model_type="t2v",
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
        **kwargs,
    ):
        # Force t2v model type for VACE compatibility
        model_type = "t2v"
        super().__init__(
            model_type,
            patch_size,
            text_len,
            in_dim,
            dim,
            ffn_dim,
            freq_dim,
            text_dim,
            out_dim,
            num_heads,
            num_layers,
            window_size,
            qk_norm,
            cross_attn_norm,
            eps,
            **kwargs,
        )

        # VACE configuration
        self.vace_layers = (
            [i for i in range(0, self.num_layers, 2)]
            if vace_layers is None
            else vace_layers
        )
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers, (
            "VACE layer 0 must be included for proper initialization"
        )
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # Replace base blocks with VACE-aware blocks
        self.blocks = nn.ModuleList(
            [
                BaseWanAttentionBlock(
                    "t2v_cross_attn",
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.window_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                    block_id=self.vace_layers_mapping[i]
                    if i in self.vace_layers
                    else None,
                    **kwargs,
                )
                for i in range(self.num_layers)
            ]
        )

        self.attach_vace = kwargs.get("attach_vace", True)

        if self.attach_vace:
            # VACE control blocks
            self.vace_blocks = nn.ModuleList(
                [
                    VaceWanAttentionBlock(
                        "t2v_cross_attn",
                        self.dim,
                        self.ffn_dim,
                        self.num_heads,
                        self.window_size,
                        self.qk_norm,
                        self.cross_attn_norm,
                        self.eps,
                        block_id=i,
                        **kwargs,
                    )
                    for i, _ in enumerate(self.vace_layers)
                ]
            )

            # VACE patch embedding for control signals
            self.vace_patch_embedding = nn.Conv3d(
                self.vace_in_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )

    def _set_gradient_checkpointing(self, *args, **kwargs):
        """
        Enable/disable gradient checkpointing for memory efficiency.
        Compatible with both old and new Diffusers API.

        Args:
            enable (bool): Whether to enable gradient checkpointing
            gradient_checkpointing_func (callable, optional): Custom checkpointing function
        """
        # Parse arguments for API compatibility
        if "enable" in kwargs or "gradient_checkpointing_func" in kwargs:
            enable = kwargs.get("enable", True)
            grad_ckpt_func = kwargs.get("gradient_checkpointing_func", None)
        elif len(args) >= 1 and isinstance(args[0], bool):
            enable = args[0]
            grad_ckpt_func = None
        else:
            enable = True
            grad_ckpt_func = None

        # Apply to all modules
        def _apply(module):
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enable

            if enable and hasattr(module, "set_gradient_checkpointing"):
                try:
                    module.set_gradient_checkpointing(grad_ckpt_func)
                except TypeError:
                    module.set_gradient_checkpointing(enable=True)

        for m in self.modules():
            _apply(m)

    def forward_vace(self, x, vace_context, seq_len, base_kwargs):
        """
        Forward pass for VACE control branch.

        Args:
            x: Main model features [B, L, C]
            vace_context: VACE control signals, list of [C_vace, F, H, W]
            seq_len: Maximum sequence length
            base_kwargs: Base forward arguments (e, seq_lens, grid_sizes, etc.)

        Returns:
            hints: List of hint features to inject into main model
        """
        # Embed VACE context
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in c
            ]
        )

        # Prepare kwargs for VACE blocks (includes main features x)
        vace_kwargs = dict(x=x)
        vace_kwargs.update(base_kwargs)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        # Process through VACE blocks
        for block in self.vace_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c,
                    **vace_kwargs,
                    use_reentrant=False,
                )
            else:
                c = block(c, **vace_kwargs)

        # Extract hints (all except the last stacked feature)
        hints = torch.unbind(c)[:-1]
        return hints

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        vace_context=None,
        vace_context_scale=1.0,
        classify_mode=False,
        concat_time_embeddings=False,
        register_tokens=None,
        cls_pred_branch=None,
        gan_ca_blocks=None,
        clip_fea=None,
        y=None,
        **kwargs,
    ):
        r"""
        Forward pass through the VACE-enhanced diffusion model.

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor] or Tensor):
                Text embeddings, either list of tensors [L, C] or pre-formatted tensor [B, L, C]
            seq_len (int):
                Maximum sequence length for positional encoding
            vace_context (List[Tensor], optional):
                VACE control signals, same format as x
            vace_context_scale (float, optional):
                Scale factor for VACE hint injection. Defaults to 1.0.
            classify_mode (bool, optional):
                Whether to run in classification mode. Defaults to False.
            concat_time_embeddings (bool, optional):
                Whether to concatenate time embeddings in classification mode
            register_tokens (nn.Module, optional):
                Register tokens for classification mode
            cls_pred_branch (nn.Module, optional):
                Classification prediction branch
            gan_ca_blocks (nn.ModuleList, optional):
                GAN cross-attention blocks for classification
            clip_fea (Tensor, optional):
                CLIP image features for image-to-video mode
            y (List[Tensor], optional):
                Conditional video inputs for image-to-video mode, same shape as x
            **kwargs:
                Additional arguments for control

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If classify_mode=False: Denoised video tensor [B, C_out, F, H/8, W/8]
                - If classify_mode=True: Tuple of (denoised videos, classification outputs)
        """
        # Device setup
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Image-to-video mode: concatenate conditional frames
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Patch embedding
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len, (
            f"Sequence length {seq_lens.max()} exceeds maximum {seq_len}"
        )
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        # Time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # Text context processing
        context_lens = None
        if isinstance(context, list):
            # Format list of variable-length embeddings
            context = self.text_embedding(
                torch.stack(
                    [
                        torch.cat(
                            [u, u.new_zeros(self.text_len - u.size(0), u.size(1))]
                        )
                        for u in context
                    ]
                )
            )
        else:
            # Pre-formatted tensor
            context = self.text_embedding(context)

        # CLIP features for image-to-video (if needed)
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # [B, 257, dim]
            context = torch.concat([context_clip, context], dim=1)

        # Prepare base forward arguments
        forward_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            action_vector=None,  # VACE model does not use action vectors
        )

        # VACE hint generation and injection
        if self.attach_vace and vace_context is not None:
            hints = self.forward_vace(x, vace_context, seq_len, forward_kwargs)
            forward_kwargs["hints"] = hints
        else:
            forward_kwargs["hints"] = None

        forward_kwargs["context_scale"] = vace_context_scale

        # Gradient checkpointing wrapper
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        # Classification mode setup
        final_x = None
        if classify_mode:
            assert register_tokens is not None, (
                "register_tokens required for classify_mode"
            )
            assert gan_ca_blocks is not None, "gan_ca_blocks required for classify_mode"
            assert cls_pred_branch is not None, (
                "cls_pred_branch required for classify_mode"
            )

            final_x = []
            from einops import repeat

            registers = repeat(register_tokens(), "n d -> b n d", b=x.shape[0])

        # Main transformer blocks
        gan_idx = 0
        for ii, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **forward_kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **forward_kwargs)

            # Classification mode feature extraction
            if classify_mode and ii in [13, 21, 29]:
                gan_token = registers[:, gan_idx : gan_idx + 1]
                final_x.append(gan_ca_blocks[gan_idx](x, gan_token))
                gan_idx += 1

        # Classification prediction
        if classify_mode:
            final_x = torch.cat(final_x, dim=1)
            if concat_time_embeddings:
                final_x = cls_pred_branch(
                    torch.cat([final_x, 10 * e[:, None, :]], dim=1).view(
                        final_x.shape[0], -1
                    )
                )
            else:
                final_x = cls_pred_branch(final_x.view(final_x.shape[0], -1))

        # Output head
        x = self.head(x, e)

        # Unpatchify to video format
        x = self.unpatchify(x, grid_sizes)

        if classify_mode:
            return torch.stack(x), final_x

        return torch.stack(x)
