"""
Video generation pipeline using latent 3d memory.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import Tensor
from torchvision.io import write_video
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from latent_mem.configs.inference_config import LatentMemPipelineConfig
from latent_mem.latent_point_cloud import LatentPointCloud
from latent_mem.wrapper.wan.bidirectonal_vace import BidirectionalWanWrapperVACE

from .utils import compute_iteration_plan, get_generator_model_type


class LatentMemPipeline:
    def __init__(
        self,
        config: LatentMemPipelineConfig,
        generator: BidirectionalWanWrapperVACE,
        vae,
        text_encoder,
        clip_encoder,
        scheduler,
    ):
        self.config = config
        self.generator = generator
        self.vae = vae
        self.text_encoder = text_encoder
        self.clip_encoder = clip_encoder
        self.scheduler = scheduler
        generator_model_type = get_generator_model_type(generator)
        self.use_image_conditioning = generator_model_type == "i2v"
        self.use_ti2v_first_frame_conditioning = generator_model_type == "ti2v"
        self.mapanything_model = None

    def generate(
        self,
        first_frame_path: str,
        text_prompt: str,
        poses_list,
        intrinsics_list,
        t0: int = 0,
        video_dir: Path | str | None = None,
        pointcloud_dir: Path | str | None = None,
    ) -> Tensor:
        """
        t0: start frame index in poses/intrinsics timeline
        poses_list: all camera poses
        intrinsics_list: all camera intrinsics
        """
        poses_list = np.asarray(poses_list)
        intrinsics_list = np.asarray(intrinsics_list)
        if intrinsics_list.shape == (3, 3):
            intrinsics_list = np.broadcast_to(intrinsics_list, (len(poses_list), 3, 3))

        from mapanything.models import MapAnything
        from mapanything.utils.image import preprocess_inputs

        runtime_device = next(self.generator.parameters()).device
        if self.mapanything_model is None:
            self.mapanything_model = MapAnything.from_pretrained(
                "facebook/map-anything"
            ).to(runtime_device)
            self.mapanything_model.eval()

        # Build latent point cloud.
        lpc = LatentPointCloud.from_image_path(
            first_frame_path,
            self.vae,
            intrinsics=intrinsics_list[t0],
            cam2world=poses_list[t0],
            device=runtime_device,
            mapanything_model=self.mapanything_model,
        )
        temporal_stride = self.vae.vae_stride[0]
        preceding_latent_frames = (
            self.config.preceding_pixel_frames + temporal_stride - 1
        ) // temporal_stride

        generated_frame_latents = []
        generated_scene_latents = []
        current_first_frame = Image.open(first_frame_path).convert("RGB")
        current_first_frame_latent = None

        if self.config.single_pass:
            iter_plan = [(0, self.config.output_frames, self.config.output_frames)]
        else:
            iter_plan = compute_iteration_plan(self.config.output_frames)
        for iter_idx, (output_start, output_end, iter_frames) in enumerate(iter_plan):
            # Build per-iteration target frame indices in pixel timeline.
            # Iter 0: [t0 ... t0+model_frames-1]
            # Later: [t0+output_start-1 ... t0+output_start-1+model_frames-1]
            # (1-frame overlap)
            if iter_idx == 0:
                pose_start = t0 + output_start
            else:
                pose_start = t0 + output_start - 1

            # Convert pixel-frame window to latent-frame window.
            num_t = (iter_frames - 1) // temporal_stride + 1
            # Match pixel-frame scene projections to the VAE latent temporal grid.(e.g. from 33 to 9) This is important.
            target_pose_indices = [
                pose_start + i * temporal_stride for i in range(num_t)
            ]
            if target_pose_indices[-1] >= len(poses_list):
                raise ValueError(
                    f"Target pose index {target_pose_indices[-1]} exceeds poses length {len(poses_list)}"
                )
            if target_pose_indices[-1] >= len(intrinsics_list):
                raise ValueError(
                    f"Target intrinsic index {target_pose_indices[-1]} exceeds intrinsics length {len(intrinsics_list)}"
                )

            # Get target scene projection latents: [T, C, H, W]
            t_scene_latents = []
            for i in target_pose_indices:
                camera_pose = torch.as_tensor(poses_list[i])
                # FIXME: Target projection currently reuses the LPC construction intrinsics.
                # camera_intrinsic = torch.as_tensor(intrinsics_list[i])
                scene_proj_latent, _ = lpc.project(camera_pose)
                t_scene_latents.append(scene_proj_latent)
            t_scene_latents = torch.stack(t_scene_latents)

            # Get Preceding and Reference context.
            if iter_idx == 0 or len(generated_frame_latents) == 0:
                p_scene_latents = None
                p_frame_latents = None
                r_frame_latents = None
            else:
                # Later iterations use the previous last frame as target overlap.
                # Preceding context must end before that overlap to match training.
                preceding_end = len(generated_frame_latents) - 1
                num_p = min(preceding_latent_frames, preceding_end)
                if num_p > 0:
                    preceding_start = preceding_end - num_p
                    p_scene_latents = torch.stack(
                        generated_scene_latents[preceding_start:preceding_end]
                    )
                    p_frame_latents = torch.stack(
                        generated_frame_latents[preceding_start:preceding_end]
                    )
                    # FIXME: This copies preceding frames as references instead of retrieving dedicated references.
                    r_frame_latents = p_frame_latents
                else:
                    p_scene_latents = None
                    p_frame_latents = None
                    r_frame_latents = None

            # Generate video (return latents)
            output_latents = self._generate_single_iter(
                first_frame=current_first_frame,
                first_frame_latent=current_first_frame_latent,
                p_latents=p_frame_latents,
                r_latents=r_frame_latents,
                p_scene_proj=p_scene_latents,
                t_scene_proj=t_scene_latents,
                text_prompt=text_prompt,
            )

            # Update states according to this iter's output
            output_latents = output_latents.squeeze(0)  # [T, C, H, W]
            current_first_frame_latent = output_latents[-1:].unsqueeze(
                0
            )  # [1, 1, C, H, W]
            t_scene_latents = t_scene_latents.to(
                device=output_latents.device, dtype=output_latents.dtype
            )

            if iter_idx == 0:
                generated_frame_latents += list(torch.unbind(output_latents, dim=0))
                generated_scene_latents += list(torch.unbind(t_scene_latents, dim=0))
                update_latents = output_latents
                update_pose_indices = target_pose_indices
            else:
                # Drop overlap frame for later iterations.
                generated_frame_latents += list(torch.unbind(output_latents[1:], dim=0))
                generated_scene_latents += list(
                    torch.unbind(t_scene_latents[1:], dim=0)
                )
                update_latents = output_latents[1:]
                update_pose_indices = target_pose_indices[1:]

            # Write newly generated views back into latent memory for later rounds.
            update_frames = []
            for latent in torch.unbind(update_latents, dim=0):
                frame = self.vae.decode_to_pixel(latent[None, None])[0, -1]
                frame = (frame + 1.0) / 2.0
                frame = frame.clamp(0, 1)
                frame = (frame * 255).byte().cpu()
                update_frames.append(rearrange(frame, "c h w -> h w c").numpy())

            views = []
            for frame, pose_idx in zip(update_frames, update_pose_indices, strict=True):
                views.append(
                    {
                        "img": frame,
                        "intrinsics": np.asarray(
                            intrinsics_list[pose_idx], dtype=np.float32
                        ),
                        "camera_poses": np.asarray(
                            poses_list[pose_idx], dtype=np.float32
                        ),
                        "is_metric_scale": torch.tensor([True]),
                    }
                )

            with torch.no_grad():
                predictions = self.mapanything_model.infer(
                    preprocess_inputs(views),
                    memory_efficient_inference=True,
                    use_amp=True,
                    amp_dtype="bf16",
                    apply_mask=True,
                    mask_edges=True,
                )

            depths = np.stack(
                [
                    pred["depth_z"].detach().cpu().numpy().squeeze(0).squeeze(-1)
                    for pred in predictions
                ],
                axis=0,
            )
            intrinsics_update = np.stack(
                [
                    pred["intrinsics"].detach().cpu().numpy().squeeze(0)
                    for pred in predictions
                ],
                axis=0,
            )
            poses_update = np.stack(
                [
                    pred["camera_poses"].detach().cpu().numpy().squeeze(0)
                    for pred in predictions
                ],
                axis=0,
            )
            lpc.update(
                depths=depths,
                intrinsics=intrinsics_update,
                cam2worlds=poses_update,
                latents=update_latents,
            )

            if self.use_image_conditioning:
                # I2V needs a pixel first-frame condition for clip and y condition.
                last_latent = output_latents[-1:].unsqueeze(0)  # [1, 1, C, H, W]
                last_frame = self.vae.decode_to_pixel(last_latent)
                last_frame = (last_frame + 1.0) / 2.0
                last_frame = last_frame.clamp(0, 1)[0, -1]
                last_frame = (last_frame * 255).byte().cpu()
                current_first_frame = Image.fromarray(
                    rearrange(last_frame, "c h w -> h w c").numpy()
                )

        # Decode all generated latents and save video
        generated_frame_latents = torch.stack(generated_frame_latents)  # [T,C,H,W]
        generated_frame_latents.unsqueeze_(0)  # [B,T,C,H,W]

        video_data = self.vae.decode_to_pixel(
            generated_frame_latents
        )  # result in [B,T,C,H,W]
        video_data.squeeze_(0)  # [T,C,H,W]
        video_data = (video_data + 1.0) / 2.0
        video_data = video_data.clamp(0, 1)
        output_frames = video_data.detach().cpu()

        if pointcloud_dir is not None:
            pointcloud_dir = Path(pointcloud_dir)
            pointcloud_dir.mkdir(parents=True, exist_ok=True)
            lpc.save(pointcloud_dir / "latent-point-cloud.pt")

        if video_dir is not None:
            video_dir = Path(video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            video_uint8 = (output_frames * 255).byte()
            video_uint8 = rearrange(video_uint8, "t c h w -> t h w c")
            output_path = video_dir / "example-latent.mp4"
            write_video(str(output_path), video_uint8, self.config.fps)

        return output_frames

    def _generate_single_iter(
        self,
        first_frame,
        first_frame_latent: Optional[Tensor],
        p_latents,
        r_latents,
        p_scene_proj,
        t_scene_proj,
        text_prompt: str,
    ):
        """
        p_latents is frame latents, while p_scene_proj is scene projection latents.
        """
        with torch.inference_mode():
            generator = self.generator
            vae = self.vae
            text_encoder = self.text_encoder
            sample_scheduler = self.scheduler

            generator_param = next(generator.parameters())
            runtime_device = generator_param.device
            runtime_dtype = generator_param.dtype

            # Prepare latents to [B, T, C, H, W]
            def _to_bfchw(latents: Optional[Tensor]):
                if latents is None:
                    return None
                if latents.ndim == 4:
                    latents = latents.unsqueeze(0)
                assert latents.ndim == 5, "Latents shape should be B,T,C,H,W"
                return latents.to(device=runtime_device, dtype=runtime_dtype)

            t_scene_proj = _to_bfchw(t_scene_proj)  # [B,T,C,H,W]
            p_scene_proj = _to_bfchw(p_scene_proj)
            p_latents = _to_bfchw(p_latents)
            r_latents = _to_bfchw(r_latents)

            batch_size, num_t, num_channels, latent_h, latent_w = t_scene_proj.shape
            num_p = 0 if p_latents is None else p_latents.shape[1]
            num_r = 0 if r_latents is None else r_latents.shape[1]

            # Prepare text and image embedding
            text_conditional_dict = text_encoder(text_prompts=[text_prompt])
            context = text_conditional_dict["prompt_embeds"].to(
                device=runtime_device, dtype=runtime_dtype
            )

            clip_fea = None
            y = None
            if self.use_image_conditioning:
                assert self.clip_encoder is not None
                img_tensor = (
                    to_tensor(first_frame)
                    .sub_(0.5)
                    .div_(0.5)
                    .to(device=runtime_device, dtype=runtime_dtype)
                )
                clip_fea = self.clip_encoder(img_tensor).to(
                    device=runtime_device, dtype=runtime_dtype
                )

                # Encode first frame with VAE (for I2V conditioning)
                # IMPORTANT: y must have total frames (T+P+R) to match model input
                total_latent_frames = num_t + num_p + num_r
                total_pixel_frames = (total_latent_frames - 1) * vae.vae_stride[0] + 1
                y = vae.run_vae_encoder(
                    img_tensor,
                    new_target_video_length=total_pixel_frames,
                )
                y = y.unsqueeze(0).to(device=runtime_device, dtype=runtime_dtype)

            if self.use_ti2v_first_frame_conditioning:
                if batch_size != 1:
                    raise ValueError(
                        "TI2V first-frame conditioning currently expects a single "
                        f"first frame, got batch_size={batch_size}."
                    )

                if first_frame_latent is None:
                    img_tensor = (
                        to_tensor(first_frame)
                        .sub_(0.5)
                        .div_(0.5)
                        .to(device=runtime_device, dtype=runtime_dtype)
                    )
                    first_frame_video = img_tensor.unsqueeze(0).unsqueeze(2)
                    first_frame_latent = vae.encode_to_latent(first_frame_video)

                first_frame_latent = first_frame_latent.to(
                    device=runtime_device, dtype=runtime_dtype
                )
                assert first_frame_latent.shape == (
                    batch_size,
                    1,
                    num_channels,
                    latent_h,
                    latent_w,
                ), (
                    "TI2V first-frame latent must have shape [B, 1, C, H, W], "
                    f"got {tuple(first_frame_latent.shape)} for expected "
                    f"{(batch_size, 1, num_channels, latent_h, latent_w)}."
                )
            else:
                first_frame_latent = None

            guidance_scale = self.config.guidance_scale

            if self.config.use_cfg:
                # FIXME: Add a negative prompt field to latent inference config.
                unconditional_dict = text_encoder(text_prompts=[""] * batch_size)
                uncon_context = unconditional_dict["prompt_embeds"].to(
                    device=runtime_device,
                    dtype=runtime_dtype,
                )
            else:
                uncon_context = None

            # Prepare VACE context [T_scene, P_scene], they are all latents.
            if p_scene_proj is not None and num_p > 0:
                vace_scene_proj = torch.cat([t_scene_proj, p_scene_proj], dim=1)
            else:
                vace_scene_proj = t_scene_proj

            vace_context = vace_scene_proj.to(
                device=runtime_device, dtype=runtime_dtype
            )  # [B,T,C,H,W]

            expected_vace_channels = None
            for module in generator.modules():
                vace_patch_embedding = getattr(module, "vace_patch_embedding", None)
                if vace_patch_embedding is not None:
                    expected_vace_channels = vace_patch_embedding.in_channels
                    break

            if expected_vace_channels == 49 and vace_context.shape[2] == 48:
                # The mask channel marks positions where all scene projection channels are zero.
                hole_mask = (~vace_context.ne(0).any(dim=2, keepdim=True)).to(
                    dtype=vace_context.dtype
                )
                vace_context = torch.cat([vace_context, hole_mask], dim=2)

            if (
                expected_vace_channels is not None
                and vace_context.shape[2] != expected_vace_channels
            ):
                raise ValueError(
                    "VACE context channel mismatch: generator expects "
                    f"{expected_vace_channels}, got {vace_context.shape[2]}."
                )

            vace_context = rearrange(vace_context, "B T C H W -> B C T H W")

            # Denosing process
            noise = torch.randn(
                batch_size,
                num_t,
                num_channels,
                latent_h,
                latent_w,
                device=runtime_device,
                dtype=runtime_dtype,
            )

            sample_scheduler.set_timesteps(self.config.infer_steps)

            latents = noise
            if first_frame_latent is not None:
                latents[:, :1].copy_(first_frame_latent)

            timestep_dtype = sample_scheduler.timesteps.dtype
            timestep_p = (
                torch.zeros(
                    [batch_size, num_p], device=runtime_device, dtype=timestep_dtype
                )
                if num_p > 0
                else None
            )
            timestep_r = (
                torch.zeros(
                    [batch_size, num_r], device=runtime_device, dtype=timestep_dtype
                )
                if num_r > 0
                else None
            )

            denoising_pbar = tqdm(
                enumerate(sample_scheduler.timesteps),
                total=len(sample_scheduler.timesteps),
                desc=f"Denoising (T={num_t}, P={num_p}, R={num_r})",
            )

            for step_idx, t in denoising_pbar:
                timestep_value = t.to(device=runtime_device, dtype=timestep_dtype)
                timestep_t = timestep_value * torch.ones(
                    [batch_size, num_t],
                    device=runtime_device,
                    dtype=timestep_dtype,
                )
                if first_frame_latent is not None:
                    timestep_t[:, 0].zero_()

                timestep_parts = [timestep_t]
                if timestep_p is not None:
                    timestep_parts.append(timestep_p)
                if timestep_r is not None:
                    timestep_parts.append(timestep_r)
                timestep = torch.cat(timestep_parts, dim=1)

                latent_parts = [latents]
                if p_latents is not None and num_p > 0:
                    latent_parts.append(p_latents)
                if r_latents is not None and num_r > 0:
                    latent_parts.append(r_latents)
                combined_latents = torch.cat(latent_parts, dim=1)

                generator_kwargs = {
                    "noisy_image_or_video": combined_latents,
                    "timestep": timestep,
                    "context": context,
                    "clip_fea": clip_fea,
                    "y": y,
                    "vace_context": vace_context,
                    "vace_context_scale": 1.0,
                    "num_t": num_t,
                    "num_p": num_p,
                    "num_r": num_r,
                }

                flow_pred_cond = generator(**generator_kwargs)

                if self.config.use_cfg and uncon_context is not None:
                    generator_kwargs["context"] = uncon_context
                    flow_pred_uncond = generator(**generator_kwargs)
                    flow_pred = flow_pred_uncond + guidance_scale * (
                        flow_pred_cond - flow_pred_uncond
                    )
                else:
                    flow_pred = flow_pred_cond

                flow_pred_t = flow_pred[:, :num_t]
                step_out = sample_scheduler.step(flow_pred_t, t, latents)
                if isinstance(step_out, tuple):
                    latents = step_out[0]
                elif hasattr(step_out, "prev_sample"):
                    latents = step_out.prev_sample
                else:
                    latents = step_out

                if first_frame_latent is not None:
                    latents[:, :1].copy_(first_frame_latent)

                denoising_pbar.set_postfix(
                    {
                        "timestep": f"{t}",
                        "step": f"{step_idx + 1}/{len(sample_scheduler.timesteps)}",
                    }
                )
            return latents
