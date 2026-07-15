from collections.abc import Iterable
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from mirage.wan2_2.modules import Wan2_1_VAE, Wan2_2_VAE


class WanVAEWrapper:
    def __init__(
        self,
        wan_model_path: Path | str,
        vae_checkpoint: str,
        device: str | torch.device = "cuda",
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
            memo=memo,
            prefix=prefix,
            remove_duplicate=remove_duplicate,
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
        self,
        pixel: Float[Tensor, "B C T H W"] | Iterable[Tensor],
    ) -> list[Tensor]:
        """
        Return a list of [C, T', h, w] tensors.

        Wan VAE handles the first frame, so this can encode a single image.
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
        if new_target_video_length is not None:
            self.target_video_length = new_target_video_length

        device, model_dtype = self._get_vae_runtime_spec()
        img = img.to(device=device, dtype=model_dtype)
        H, W = img.shape[1:]

        if not add_first_to_kvcache:
            vae_encode_out = self.encode(
                [
                    torch.concat(
                        [
                            img.unsqueeze(1),
                            torch.zeros(
                                3,
                                self.target_video_length - 1,
                                H,
                                W,
                                device=img.device,
                                dtype=model_dtype,
                            ),
                        ],
                        dim=1,
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
        )

        if not add_first_to_kvcache:
            msk[:, 1:] = 0
        else:
            msk[:, 0:] = 0

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
            dim=1,
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, h, w)
        msk = msk.transpose(1, 2)[0]

        return torch.concat([msk, vae_encode_out]).to(self.dtype)

    def run_vae_encoder_batch(
        self,
        imgs: Float[Tensor, "B 3 H W"],
        new_target_video_length=None,
        add_first_to_kvcache=False,
    ):
        if new_target_video_length is not None:
            self.target_video_length = new_target_video_length

        device, model_dtype = self._get_vae_runtime_spec()
        imgs = imgs.to(device=device, dtype=model_dtype)
        B, _, H, W = imgs.shape

        if not add_first_to_kvcache:
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

        msk = torch.ones(
            B,
            self.target_video_length,
            h,
            w,
            device=imgs.device,
            dtype=model_dtype,
        )

        if not add_first_to_kvcache:
            msk[:, 1:] = 0
        else:
            msk[:, 0:] = 0

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
            dim=1,
        )
        msk = msk.view(B, msk.shape[1] // 4, 4, h, w)
        msk = msk.transpose(1, 2)

        vae_encode_out = torch.stack(vae_encode_out, dim=0)
        return torch.concat([msk, vae_encode_out], dim=1).to(self.dtype)

    def encode_to_latent(
        self,
        pixel: Float[Tensor, "B C T H W"],
    ) -> Float[Tensor, "B T C H W"]:
        output = torch.stack(self.encode(pixel))
        return output.permute(0, 2, 1, 3, 4)

    def decode_to_pixel(
        self,
        latent: Float[Tensor, "B T C H W"],
    ) -> Float[Tensor, "B T C H W"]:
        zs = latent.permute(0, 2, 1, 3, 4)

        device, model_dtype = self._get_vae_runtime_spec()
        latent_list = [u.to(device=device, dtype=model_dtype) for u in zs]

        output = torch.stack(self._vae.decode(latent_list))
        return output.permute(0, 2, 1, 3, 4)
