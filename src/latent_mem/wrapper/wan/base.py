import os
from pathlib import Path
from typing import Iterable, List

import torch
from jaxtyping import Float
from torch import Tensor

from latent_mem.wan.modules.clip import CLIPModel
from latent_mem.wan.modules.t5 import umt5_xxl
from latent_mem.wan.modules.tokenizers import HuggingfaceTokenizer
from latent_mem.wan2_2.modules import Wan2_1_VAE, Wan2_2_VAE


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name

        self.text_encoder = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )
            .eval()
            .requires_grad_(False)
        )

        # Dynamically construct path based on model_name
        t5_encoder_path = os.path.join(model_name, "models_t5_umt5-xxl-enc-bf16.pth")
        self.text_encoder.load_state_dict(
            torch.load(t5_encoder_path, map_location="cpu", weights_only=False)
        )

        # Dynamically construct tokenizer path
        tokenizer_path = os.path.join(model_name, "google/umt5-xxl/")
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean="whitespace"
        )

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True
        )
        # Use the device of the model's parameters instead of hardcoded device
        device = next(self.text_encoder.parameters()).device
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {"prompt_embeds": context}


class WanVAEWrapper:
    def __init__(
        self,
        wan_model_path: Path | str,
        vae_checkpoint: str,
        device="cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.dtype = dtype
        self.device = device
        self.vae_checkpoint = vae_checkpoint

        vae_path = Path(wan_model_path) / vae_checkpoint
        self._vae = self._build_vae(vae_path)
        self.model = self._vae.model
        self.mean = self._vae.scale[0]
        self.std = 1.0 / self._vae.scale[1]

        self.vae_stride = (4, 16, 16) if self._is_wan_2_2_checkpoint() else (4, 8, 8)
        self.target_video_length = 81

    def _is_wan_2_2_checkpoint(self) -> bool:
        return "2.2" in self.vae_checkpoint

    def _build_vae(self, vae_path: Path):
        if self._is_wan_2_2_checkpoint():
            return Wan2_2_VAE(vae_pth=vae_path, dtype=self.dtype, device=self.device)
        return Wan2_1_VAE(vae_pth=vae_path, dtype=self.dtype, device=self.device)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def modules(self):
        return self.model.modules()

    def named_modules(
        self,
        memo=None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        return self.model.named_modules(
            memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
        )

    def children(self):
        return self.model.children()

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)

        runtime_dtype = kwargs.get("dtype")
        if runtime_dtype is None:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    runtime_dtype = arg
                    break
        if runtime_dtype is not None:
            self.dtype = runtime_dtype
            self.mean = self.mean.to(dtype=runtime_dtype)
            self.std = self.std.to(dtype=runtime_dtype)

        runtime_device = kwargs.get("device")
        if runtime_device is None:
            for arg in args:
                if isinstance(arg, (torch.device, str)):
                    runtime_device = arg
                    break
        if runtime_device is not None:
            self.device = runtime_device
            self.mean = self.mean.to(device=runtime_device)
            self.std = self.std.to(device=runtime_device)

        self._vae.model = self.model
        self._vae.scale = [self.mean, 1.0 / self.std]
        self._vae.dtype = self.dtype
        self._vae.device = self.device
        return self

    def _get_vae_runtime_spec(self):
        param = next(self.model.parameters())
        return param.device, param.dtype

    @torch.inference_mode()
    def encode(
        self, pixel: Float[Tensor, "B C T H W"] | Iterable[Tensor]
    ) -> List[Tensor]:
        """
        Return shape: list of [C,T',h,w]. WanVAE will handle first frame, so this fucntion can be used to encode a single image.
        """
        if isinstance(pixel, torch.Tensor):
            assert pixel.ndim in [4, 5], (
                f"Expected video tensor with 4 or 5 dims, got shape {tuple(pixel.shape)}"
            )
            if pixel.ndim == 4:
                videos = [pixel]
            else:
                videos = list(pixel.unbind())
        else:
            videos = list(pixel)

        device, model_dtype = self._get_vae_runtime_spec()
        video_list = [video.to(device=device, dtype=model_dtype) for video in videos]
        return self._vae.encode(video_list)

    def run_vae_encoder(
        self,
        img: Float[Tensor, "3 H W"],
        new_target_video_length=None,
        add_first_to_kvcache=False,
    ):
        # NOTE image don't need to repeat 4 times because Wan VAE handles this case. But msk need to repeat in order to align with vae outputs.
        if new_target_video_length is not None:
            self.target_video_length = new_target_video_length

        device, model_dtype = self._get_vae_runtime_spec()
        img = img.to(device=device, dtype=model_dtype)
        H, W = img.shape[1:]

        if not add_first_to_kvcache:
            # For standard I2V
            vae_encode_out = self.encode(
                [
                    torch.concat(
                        [
                            img.unsqueeze(1),  # [3, 1, H, W]
                            torch.zeros(
                                3,
                                self.target_video_length - 1,
                                H,
                                W,
                                device=img.device,
                                dtype=model_dtype,
                            ),  # [3, L-1, H, W]
                        ],
                        dim=1,  # [3, L, H, W]
                    )
                ],
            )[0]
        else:
            vae_encode_out = self.encode(
                [
                    torch.zeros(
                        3,
                        self.target_video_length,
                        H,
                        W,
                        device=img.device,
                        dtype=model_dtype,
                    )
                ],
            )[0]

        h = H // self.vae_stride[1]
        w = W // self.vae_stride[2]
        msk = torch.ones(
            1,
            self.target_video_length,
            h,
            w,
            device=img.device,
            dtype=model_dtype,
        )  # [1,L,h,w]

        if not add_first_to_kvcache:
            # For standard I2V, we don't need to generate the first frame.
            msk[:, 1:] = 0
        else:
            # In this case, we need to generate the first frame by model.
            msk[:, 0:] = 0

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )  # Repeat first frame 4 times
        msk = msk.view(1, msk.shape[1] // 4, 4, h, w)
        msk = msk.transpose(1, 2)[0]  # [4,(L+3)/4,h,w], 4 is the channel dim.

        vae_encode_out = torch.concat([msk, vae_encode_out]).to(self.dtype)
        # FIXME: The shape comment below is only correct for 16-channel Wan2.1
        # latents plus 4 mask channels. Wan2.2 produces 48 latent channels, so
        # the returned channel count is different even though the code path works.
        return vae_encode_out  # [20, 21, h, w]

    def run_vae_encoder_batch(
        self,
        imgs: Float[Tensor, "B 3 H W"],
        new_target_video_length=None,
        add_first_to_kvcache=False,
    ):
        """
        Batch version of run_vae_encoder that supports batched image input.
        """
        if new_target_video_length is not None:
            self.target_video_length = new_target_video_length

        device, model_dtype = self._get_vae_runtime_spec()
        imgs = imgs.to(device=device, dtype=model_dtype)
        B, _, H, W = imgs.shape

        if not add_first_to_kvcache:
            # For standard I2V
            # Create input: [B, 3, target_video_length, H, W]
            video_inputs = []
            for i in range(B):
                video_input = torch.concat(
                    [
                        imgs[i].unsqueeze(1),  # [3, 1, H, W]
                        torch.zeros(
                            3,
                            self.target_video_length - 1,
                            H,
                            W,
                            device=imgs.device,
                            dtype=model_dtype,
                        ),  # [3, L-1, H, W]
                    ],
                    dim=1,  # [3, L, H, W]
                )
                video_inputs.append(video_input)

            video_inputs = torch.concat(
                [
                    imgs.unsqueeze(2),
                    torch.zeros(
                        B,
                        3,
                        self.target_video_length - 1,
                        H,
                        W,
                        device=imgs.device,
                        dtype=model_dtype,
                    ),
                ],
                dim=2,
            )

            vae_encode_out = self.encode(video_inputs)
        else:
            video_inputs = torch.zeros(
                B,
                3,
                self.target_video_length,
                H,
                W,
                device=imgs.device,
                dtype=model_dtype,
            )
            vae_encode_out = self.encode(video_inputs)

        h = H // self.vae_stride[1]
        w = W // self.vae_stride[2]

        # Create mask for each batch element
        msk = torch.ones(
            B,
            self.target_video_length,
            h,
            w,
            device=imgs.device,
            dtype=model_dtype,
        )  # [B, L, h, w]

        if not add_first_to_kvcache:
            # For standard I2V, we don't need to generate the first frame.
            msk[:, 1:] = 0
        else:
            # In this case, we need to generate the first frame by model.
            msk[:, 0:] = 0

        # Process mask: repeat first frame 4 times
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )  # [B, L+3, h, w]
        msk = msk.view(B, msk.shape[1] // 4, 4, h, w)  # [B, (L+3)/4, 4, h, w]
        msk = msk.transpose(1, 2)  # [B, 4, (L+3)/4, h, w]

        # Stack VAE outputs and concatenate with masks
        # FIXME: The shape comments in this block still assume 16 latent channels.
        # Wan2.2 uses 48 latent channels, so these comments are misleading.
        vae_encode_out = torch.stack(vae_encode_out, dim=0)  # [B, 16, L, h, w]
        output = torch.concat([msk, vae_encode_out], dim=1).to(
            self.dtype
        )  # [B, 20, 21, h, w]

        return output

    def encode_to_latent(
        self, pixel: Float[Tensor, "B C T H W"]
    ) -> Float[Tensor, "B T C H W"]:
        output = torch.stack(self.encode(pixel))
        output = output.permute(0, 2, 1, 3, 4)  # from [B,C,T,H,W] to [B,T,C,H,W]
        return output

    def decode_to_pixel(
        self, latent: Float[Tensor, "B T C H W"]
    ) -> Float[Tensor, "B T C H W"]:
        zs = latent.permute(0, 2, 1, 3, 4)  # got [B,C,T,H,W]

        device, model_dtype = self._get_vae_runtime_spec()
        latent_list = [u.to(device=device, dtype=model_dtype) for u in zs]

        output = torch.stack(self._vae.decode(latent_list))  # got [B,C,T,H,W]
        output = output.permute(0, 2, 1, 3, 4)  # got [B,T,C,H,W]
        return output


class WanCLIPEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name="hf_cache/Wan-AI--Wan2.1-T2V-1.3B",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_name = model_name
        self.image_encoder = CLIPModel(
            dtype=dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(
                f"{self.model_name}/",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
        )

    @property
    def current_dtype(self):
        return next(self.image_encoder.parameters()).dtype

    def forward(self, img):
        if img.ndim == 3:
            img = img[:, None, :, :]
        elif img.ndim == 4:
            img = img.transpose(0, 1)

        img = img.to(dtype=self.current_dtype)
        clip_encoder_out = self.image_encoder.visual([img]).squeeze(0)
        return clip_encoder_out
