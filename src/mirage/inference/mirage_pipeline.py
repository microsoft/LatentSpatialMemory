from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torch import Tensor
from tqdm.auto import tqdm

from mirage.inference.utils import compute_iteration_plan
from mirage.latent_point_cloud import LatentPointCloud
from mirage.spatia.flow_match import FlowUniPCMultistepScheduler
from mirage.spatia.utils import load_state_dict
from mirage.spatia.wan_video_new import WanVideoPipeline, model_fn_wan_video


@dataclass
class VideoGeometry:
    frames: np.ndarray
    depths: np.ndarray
    intrinsics: np.ndarray
    poses_c2w: np.ndarray
    masks: np.ndarray | None = None
    frame_indices: np.ndarray | None = None
    original_size: tuple[int, int] | None = None
    processed_size: tuple[int, int] | None = None


@dataclass
class MirageConfig:
    num_frames: int = 33
    start_frame: int = 0
    infer_steps: int = 20
    num_train_timesteps: int = 1000
    timestep_shift: float = 5.0
    guidance_scale: float = 1.0
    no_cfg: bool = True
    fps: int = 16
    max_reference_frames: int = 4
    preceding_pixel_frames: int = 8
    seed: int = 42
    height: int | None = None
    width: int | None = None
    tiled: bool = True
    tile_size: tuple[int, int] = (30, 52)
    tile_stride: tuple[int, int] = (15, 26)
    mapanything_model_id: str = "facebook/map-anything"
    ref_iou_threshold: float = 0.04
    ref_iou_voxel_size: float = 0.1


def load_vace_checkpoint(pipe: WanVideoPipeline, path: Path) -> None:
    checkpoint = open_checkpoint(path)
    state = extract_vace_state_dict(checkpoint)
    if not state:
        raise ValueError(f"No VACE tensors found in {path}.")

    target_state = pipe.vace.state_dict()
    compatible = {}
    skipped = 0
    for key, value in state.items():
        if key not in target_state or value.shape != target_state[key].shape:
            skipped += 1
            continue
        compatible[key] = value.to(dtype=target_state[key].dtype)

    missing, unexpected = pipe.vace.load_state_dict(compatible, strict=False)
    print(
        "Loaded VACE checkpoint: "
        f"{path} tensors={len(compatible)} skipped={skipped} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def open_checkpoint(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        return dict(load_safetensors(str(path), device="cpu"))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict: {path}")
    return checkpoint


def extract_vace_state_dict(checkpoint: dict[str, Any]) -> dict[str, Tensor]:
    candidate = checkpoint
    if isinstance(checkpoint.get("vace"), dict):
        candidate = checkpoint["vace"]
    elif isinstance(checkpoint.get("generator"), dict) and isinstance(
        checkpoint["generator"].get("vace"), dict
    ):
        candidate = checkpoint["generator"]["vace"]
    elif isinstance(checkpoint.get("state_dict"), dict):
        candidate = checkpoint["state_dict"]

    state = {}
    for key, value in candidate.items():
        if not torch.is_tensor(value):
            continue
        normalized = strip_prefixes(
            key,
            ("module.", "model.", "pipe.vace.", "vace."),
        )
        if normalized.startswith(("vace_blocks.", "vace_patch_embedding.")):
            state[normalized] = value
    return state


def strip_prefixes(key: str, prefixes: tuple[str, ...]) -> str:
    normalized = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def validate_pipe(pipe: WanVideoPipeline) -> None:
    if pipe.dit is None:
        raise ValueError("WanVideoPipeline did not load a DiT model.")
    if pipe.vace is None:
        raise ValueError("WanVideoPipeline did not load a VACE model.")
    if not getattr(pipe.dit, "fuse_vae_embedding_in_latents", False):
        raise ValueError()
    if not getattr(pipe.dit, "seperated_timestep", False):
        raise ValueError()

    patch_embedding = getattr(pipe.vace, "vace_patch_embedding", None)
    if patch_embedding is None:
        raise ValueError()
    if int(patch_embedding.in_channels) != 96:
        raise ValueError()


class MiragePipeline:
    def __init__(
        self,
        pipe: WanVideoPipeline,
        config: MirageConfig,
    ) -> None:
        self.pipe = pipe
        self.config = config
        self.mapanything_model = None

    @torch.inference_mode()
    def generate(
        self,
        *,
        geometry_path: Path,
        prompt: str,
        output_dir: Path,
        run_metadata: dict[str, Any] | None = None,
    ) -> Tensor:
        geometry = load_video_geometry_for_inference(
            geometry_path,
            start_frame=self.config.start_frame,
        )
        validate_geometry(geometry, start_frame=self.config.start_frame)

        output_dir.mkdir(parents=True, exist_ok=True)
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device(self.pipe.device)
        dtype = self.pipe.torch_dtype
        height, width = resolve_output_hw(geometry, self.config)
        temporal_stride = 4
        preceding_latent_frames = (
            self.config.preceding_pixel_frames + temporal_stride - 1
        ) // temporal_stride
        iteration_plan = compute_iteration_plan(self.config.num_frames)

        self.pipe.load_models_to_device(["vae"])
        first_frame = resize_frame(
            geometry.frames[self.config.start_frame],
            height,
            width,
        )
        first_frame_latent = encode_video_frames(
            self.pipe,
            first_frame[None],
            tiled=self.config.tiled,
            tile_size=self.config.tile_size,
            tile_stride=self.config.tile_stride,
        ).to(device=device, dtype=dtype)

        initial_mask = get_initial_exclusion_mask(geometry, self.config.start_frame)
        lpc = LatentPointCloud.from_geometry(
            depth=geometry.depths[self.config.start_frame],
            intrinsics=geometry.intrinsics[self.config.start_frame],
            cam2world=geometry.poses_c2w[self.config.start_frame],
            latent=first_frame_latent[0],
            mask=initial_mask,
            device=device,
        )

        context = self.encode_prompt(prompt, positive=True)
        uncon_context = None
        if not self.config.no_cfg:
            uncon_context = self.encode_prompt("", positive=False)

        generator = torch.Generator(device=device)
        generator.manual_seed(self.config.seed)

        generated_latents: list[Tensor] = []
        generated_scene_latents: list[Tensor] = []
        generated_frames: list[np.ndarray] = []
        frame_visible_points: dict[int, np.ndarray] = {}
        metadata: dict[str, Any] = {
            "config": asdict(self.config),
            "geometry_path": str(geometry_path),
            "height": height,
            "width": width,
            "iterations": [],
        }
        if run_metadata is not None:
            metadata.update(run_metadata)

        for iter_idx, (output_start, output_end, model_frames) in enumerate(
            iteration_plan
        ):
            target_pose_indices = build_target_pose_indices(
                start_frame=self.config.start_frame,
                output_start=output_start,
                model_frames=model_frames,
                temporal_stride=temporal_stride,
                iter_idx=iter_idx,
            )
            if target_pose_indices[-1] >= len(geometry.poses_c2w):
                raise ValueError(
                    "Target pose index exceeds geometry length: "
                    f"{target_pose_indices[-1]} >= {len(geometry.poses_c2w)}"
                )

            target_scene = project_lpc_sequence(
                lpc=lpc,
                geometry=geometry,
                frame_indices=target_pose_indices,
            )
            preceding_latents, preceding_scene = select_preceding_context(
                generated_latents=generated_latents,
                generated_scene_latents=generated_scene_latents,
                num_frames=preceding_latent_frames,
            )
            reference_latents, reference_indices = select_reference_latents(
                lpc=lpc,
                geometry=geometry,
                target_pose_indices=target_pose_indices,
                generated_latents=generated_latents,
                frame_visible_points=frame_visible_points,
                max_reference_frames=self.config.max_reference_frames,
                iou_threshold=self.config.ref_iou_threshold,
                voxel_size=self.config.ref_iou_voxel_size,
            )

            if iter_idx == 0:
                iter_first_latent = first_frame_latent
            else:
                iter_first_latent = generated_latents[-1][None, :, None]

            output_latents = self.generate_single_iteration(
                target_scene=target_scene,
                first_frame_latent=iter_first_latent,
                preceding_latents=preceding_latents,
                preceding_scene=preceding_scene,
                reference_latents=reference_latents,
                context=context,
                uncon_context=uncon_context,
                generator=generator,
            )

            iter_video = decode_latents_to_uint8(
                self.pipe,
                output_latents,
                tiled=self.config.tiled,
                tile_size=self.config.tile_size,
                tile_stride=self.config.tile_stride,
            )
            iter_video_path = (
                video_dir
                / f"iteration_{iter_idx + 1:02d}_frames{output_start}-{output_end - 1}.mp4"
            )
            write_iteration_video(
                iter_video_path, iter_video, iter_idx, self.config.fps
            )

            output_latents_tchw = rearrange(output_latents[0], "c t h w -> t c h w")
            target_scene_tchw = target_scene.to(
                device=output_latents_tchw.device,
                dtype=output_latents_tchw.dtype,
            )

            if iter_idx == 0:
                new_latents = output_latents_tchw
                new_scene = target_scene_tchw
                new_pose_indices = target_pose_indices
                new_images = select_latent_aligned_frames(iter_video, temporal_stride)
                generated_frames.extend(list(iter_video))
            else:
                new_latents = output_latents_tchw[1:]
                new_scene = target_scene_tchw[1:]
                new_pose_indices = target_pose_indices[1:]
                new_images = select_latent_aligned_frames(
                    iter_video,
                    temporal_stride,
                )[1:]
                generated_frames.extend(list(iter_video[1:]))

            generated_latents.extend(list(torch.unbind(new_latents.detach(), dim=0)))
            generated_scene_latents.extend(
                list(torch.unbind(new_scene.detach(), dim=0))
            )

            self.update_latent_memory(
                lpc=lpc,
                geometry=geometry,
                images=new_images,
                pose_indices=new_pose_indices,
                latents=new_latents,
            )
            update_frame_visibility(
                lpc=lpc,
                geometry=geometry,
                pose_indices=new_pose_indices,
                frame_visible_points=frame_visible_points,
                start_output_latent=len(generated_latents) - len(new_pose_indices),
            )

            metadata["iterations"].append(
                {
                    "iteration": iter_idx + 1,
                    "output_start": output_start,
                    "output_end": output_end,
                    "model_frames": model_frames,
                    "target_pose_indices": target_pose_indices,
                    "reference_indices": reference_indices,
                    "num_preceding_latents": 0
                    if preceding_latents is None
                    else int(preceding_latents.shape[2]),
                    "num_reference_latents": 0
                    if reference_latents is None
                    else int(reference_latents.shape[2]),
                }
            )

        final_video = np.stack(generated_frames, axis=0)[: self.config.num_frames]
        final_path = video_dir / "generated.mp4"
        write_mp4(final_path, final_video, self.config.fps)

        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

        return torch.from_numpy(final_video)

    def encode_prompt(self, prompt: str, *, positive: bool) -> Tensor:
        self.pipe.load_models_to_device(["text_encoder"])
        return self.pipe.prompter.encode_prompt(
            prompt,
            positive=positive,
            device=self.pipe.device,
        ).to(device=self.pipe.device, dtype=self.pipe.torch_dtype)

    def generate_single_iteration(
        self,
        *,
        target_scene: Tensor,
        first_frame_latent: Tensor,
        preceding_latents: Tensor | None,
        preceding_scene: Tensor | None,
        reference_latents: Tensor | None,
        context: Tensor,
        uncon_context: Tensor | None,
        generator: torch.Generator,
    ) -> Tensor:
        device = torch.device(self.pipe.device)
        dtype = self.pipe.torch_dtype
        channels, num_t, latent_h, latent_w = target_scene.shape
        num_p = 0 if preceding_latents is None else preceding_latents.shape[2]
        num_r = 0 if reference_latents is None else reference_latents.shape[2]

        noise = torch.randn(
            (1, channels, num_t, latent_h, latent_w),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        latents_t = noise
        latents_t[:, :, :1].copy_(first_frame_latent.to(device=device, dtype=dtype))

        vace_context = build_vace_context_96(
            pipe=self.pipe,
            target_scene=target_scene,
            preceding_scene=preceding_scene,
            dtype=dtype,
            device=device,
            tiled=self.config.tiled,
            tile_size=self.config.tile_size,
            tile_stride=self.config.tile_stride,
        )
        self.pipe.load_models_to_device(["dit", "vace"])

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            shift=1.0,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(
            self.config.infer_steps,
            device=device,
            shift=self.config.timestep_shift,
        )

        timestep_p = (
            torch.zeros(num_p, device=device, dtype=torch.float32)
            if num_p > 0
            else None
        )
        timestep_r = (
            torch.zeros(num_r, device=device, dtype=torch.float32)
            if num_r > 0
            else None
        )

        for step_idx, timestep in enumerate(
            tqdm(
                scheduler.timesteps,
                desc=f"Denoising (T={num_t}, P={num_p}, R={num_r})",
            )
        ):
            timestep_t = timestep.to(device=device, dtype=torch.float32).repeat(num_t)
            timestep_t[0] = 0
            timestep_parts = []
            if timestep_r is not None:
                timestep_parts.append(timestep_r)
            timestep_parts.append(timestep_t)
            if timestep_p is not None:
                timestep_parts.append(timestep_p)
            full_timestep = torch.cat(timestep_parts, dim=0)

            latent_parts = []
            if reference_latents is not None and num_r > 0:
                latent_parts.append(reference_latents.to(device=device, dtype=dtype))
            latent_parts.append(latents_t)
            if preceding_latents is not None and num_p > 0:
                latent_parts.append(preceding_latents.to(device=device, dtype=dtype))
            combined_latents = torch.cat(latent_parts, dim=2)

            model_kwargs = {
                "dit": self.pipe.dit,
                "vace": self.pipe.vace,
                "latents": combined_latents,
                "timestep": full_timestep,
                "context": context,
                "vace_context": vace_context,
                "vace_scale": 1.0,
                "num_ref_frames": num_r,
                "fuse_vae_embedding_in_latents": True,
            }
            if step_idx == 0:
                print(
                    f"latents={tuple(combined_latents.shape)} "
                    f"vace_context={(1, *tuple(vace_context[0].shape))} "
                    f"first_timestep={float(full_timestep[num_r].item())}"
                )
            flow_cond = model_fn_wan_video(**model_kwargs)

            if not self.config.no_cfg and uncon_context is not None:
                model_kwargs["context"] = uncon_context
                flow_uncond = model_fn_wan_video(**model_kwargs)
                flow = flow_uncond + self.config.guidance_scale * (
                    flow_cond - flow_uncond
                )
            else:
                flow = flow_cond

            flow_t = flow[:, :, num_r : num_r + num_t]
            latents_t = scheduler.step(
                flow_t,
                timestep,
                latents_t,
                return_dict=False,
                generator=generator,
            )[0]
            latents_t[:, :, :1].copy_(first_frame_latent.to(device=device, dtype=dtype))

        return latents_t

    def update_latent_memory(
        self,
        *,
        lpc: LatentPointCloud,
        geometry: VideoGeometry,
        images: np.ndarray,
        pose_indices: list[int],
        latents: Tensor,
    ) -> None:
        if len(images) == 0:
            return

        predictions = infer_mapanything_depths(
            images=images,
            geometry=geometry,
            pose_indices=pose_indices,
            model_id=self.config.mapanything_model_id,
            device=torch.device(self.pipe.device),
            model_cache=self,
        )
        depths = np.stack([item["depth"] for item in predictions], axis=0)
        depth_hw = depths.shape[1:3]
        intrinsics = np.stack(
            [
                scale_intrinsics_to_hw(
                    geometry.intrinsics[pose_idx],
                    geometry.frames.shape[1:3],
                    depth_hw,
                )
                for pose_idx in pose_indices
            ],
            axis=0,
        )
        poses = geometry.poses_c2w[np.asarray(pose_indices)]
        lpc.update(
            depths=depths,
            intrinsics=intrinsics,
            cam2worlds=poses,
            latents=latents,
        )


def load_video_geometry_for_inference(
    path: Path,
    *,
    start_frame: int,
) -> VideoGeometry:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        frames = np.asarray(data["frames"]) if "frames" in data.files else None
        depths = np.asarray(data["depths"], dtype=np.float32)
        intrinsics = np.asarray(data["intrinsics"])
        poses_c2w = np.asarray(data["poses_c2w"])
        masks = load_optional_array(data, "masks")
        frame_indices = load_optional_array(data, "frame_indices")
        original_size = load_optional_hw(data, "original_size")
        processed_size = load_optional_hw(data, "processed_size")

    if frames is None:
        frames = load_geometry_rgb_frames(
            sample_dir=path.parent,
            processed_size=processed_size,
            start_frame=start_frame,
        )

    return VideoGeometry(
        frames=frames,
        depths=depths,
        intrinsics=intrinsics,
        poses_c2w=poses_c2w,
        masks=masks,
        frame_indices=frame_indices,
        original_size=original_size,
        processed_size=processed_size,
    )


def load_optional_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    value = np.asarray(data[key])
    if value.size == 0:
        return None
    return value


def load_optional_hw(data: np.lib.npyio.NpzFile, key: str) -> tuple[int, int] | None:
    value = load_optional_array(data, key)
    if value is None:
        return None
    hw = tuple(int(x) for x in value.tolist())
    if len(hw) != 2 or hw == (-1, -1):
        return None
    return hw


def load_geometry_rgb_frames(
    *,
    sample_dir: Path,
    processed_size: tuple[int, int] | None,
    start_frame: int,
) -> np.ndarray:
    clip_path = sample_dir / "clip.mp4"
    if clip_path.exists():
        return load_video_rgb_frames(clip_path, target_hw=processed_size)

    first_frame_path = sample_dir / "first_frame.png"
    if first_frame_path.exists():
        if start_frame != 0:
            raise ValueError(
                "geometry.npz has no frames and only first_frame.png is available; "
                f"cannot recover start_frame={start_frame}."
            )
        image = Image.open(first_frame_path).convert("RGB")
        if processed_size is not None:
            image = image.resize(
                (processed_size[1], processed_size[0]),
                Image.Resampling.BILINEAR,
            )
        return np.asarray(image, dtype=np.uint8)[None]

    raise FileNotFoundError(
        "geometry.npz has no frames. Expected clip.mp4 or first_frame.png in "
        f"{sample_dir}."
    )


def load_video_rgb_frames(
    path: Path,
    *,
    target_hw: tuple[int, int] | None,
) -> np.ndarray:
    import imageio.v2 as imageio

    frames = []
    reader = None
    try:
        reader = imageio.get_reader(str(path), format="FFMPEG")
        for frame in reader:
            image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
            if target_hw is not None:
                image = image.resize(
                    (target_hw[1], target_hw[0]),
                    Image.Resampling.BILINEAR,
                )
            frames.append(np.asarray(image, dtype=np.uint8))
    finally:
        if reader is not None:
            reader.close()
    if not frames:
        raise RuntimeError(f"No frames loaded from video: {path}")
    return np.stack(frames, axis=0)


def validate_geometry(geometry: VideoGeometry, *, start_frame: int) -> None:
    required = {
        "frames": geometry.frames,
        "depths": geometry.depths,
        "intrinsics": geometry.intrinsics,
        "poses_c2w": geometry.poses_c2w,
    }
    for name, value in required.items():
        if value is None:
            raise ValueError(f"geometry.npz is missing {name}.")
    if len(geometry.depths) != len(geometry.poses_c2w):
        raise ValueError("geometry depths/poses_c2w length mismatch.")
    if len(geometry.intrinsics) != len(geometry.poses_c2w):
        raise ValueError("geometry intrinsics/poses_c2w length mismatch.")
    if len(geometry.frames) not in {1, len(geometry.poses_c2w)}:
        raise ValueError(
            "geometry RGB frames must contain either the first frame or the full "
            "pose sequence."
        )
    if start_frame >= len(geometry.frames):
        raise ValueError(
            f"RGB frames contain {len(geometry.frames)} frame(s), cannot read "
            f"start_frame={start_frame}."
        )


def resolve_output_hw(
    geometry: VideoGeometry,
    config: MirageConfig,
) -> tuple[int, int]:
    if config.height is not None or config.width is not None:
        if config.height is None or config.width is None:
            raise ValueError("--height and --width must be provided together.")
        return int(config.height), int(config.width)
    if geometry.original_size is not None:
        return int(geometry.original_size[0]), int(geometry.original_size[1])
    return int(geometry.frames.shape[1]), int(geometry.frames.shape[2])


def get_initial_exclusion_mask(
    geometry: VideoGeometry,
    frame_idx: int,
) -> np.ndarray | None:
    if geometry.masks is None:
        return None
    return ~geometry.masks[frame_idx].astype(bool)


def resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    image = Image.fromarray(frame.astype(np.uint8))
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))


def encode_video_frames(
    pipe: WanVideoPipeline,
    frames: np.ndarray,
    *,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
) -> Tensor:
    vae_dtype = next(pipe.vae.parameters()).dtype
    video = torch.from_numpy(frames).float()
    video = rearrange(video, "t h w c -> c t h w").div(127.5).sub(1.0)
    video = video.to(dtype=vae_dtype)
    return pipe.vae.encode(
        [video],
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )


def decode_latents_to_uint8(
    pipe: WanVideoPipeline,
    latents: Tensor,
    *,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
) -> np.ndarray:
    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(
        latents,
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    video = video[0].add(1.0).mul(0.5).clamp(0.0, 1.0)
    video = rearrange(video, "c t h w -> t h w c")
    return video.mul(255).byte().cpu().numpy()


def build_target_pose_indices(
    *,
    start_frame: int,
    output_start: int,
    model_frames: int,
    temporal_stride: int,
    iter_idx: int,
) -> list[int]:
    if iter_idx == 0:
        pose_start = start_frame + output_start
    else:
        pose_start = start_frame + output_start - 1
    num_latent_frames = (model_frames - 1) // temporal_stride + 1
    return [pose_start + i * temporal_stride for i in range(num_latent_frames)]


def project_lpc_sequence(
    *,
    lpc: LatentPointCloud,
    geometry: VideoGeometry,
    frame_indices: list[int],
) -> Tensor:
    latents = []
    for frame_idx in frame_indices:
        intrinsics_latent = scale_intrinsics_to_hw(
            geometry.intrinsics[frame_idx],
            geometry.frames.shape[1:3],
            lpc.latent_hw,
        )
        latent, _ = lpc.project(
            cam2world=geometry.poses_c2w[frame_idx],
            intrinsics=intrinsics_latent,
        )
        latents.append(latent)
    return torch.stack(latents, dim=1)


def select_preceding_context(
    *,
    generated_latents: list[Tensor],
    generated_scene_latents: list[Tensor],
    num_frames: int,
) -> tuple[Tensor | None, Tensor | None]:
    if len(generated_latents) <= 1 or num_frames <= 0:
        return None, None
    stop = len(generated_latents) - 1
    start = max(0, stop - num_frames)
    if start == stop:
        return None, None
    latents = torch.stack(generated_latents[start:stop], dim=1).unsqueeze(0)
    scene = torch.stack(generated_scene_latents[start:stop], dim=1)
    return latents, scene


def select_reference_latents(
    *,
    lpc: LatentPointCloud,
    geometry: VideoGeometry,
    target_pose_indices: list[int],
    generated_latents: list[Tensor],
    frame_visible_points: dict[int, np.ndarray],
    max_reference_frames: int,
    iou_threshold: float,
    voxel_size: float,
) -> tuple[Tensor | None, list[int]]:
    if max_reference_frames <= 0 or not generated_latents or not frame_visible_points:
        return None, []

    target_points = []
    for frame_idx in target_pose_indices:
        target_points.append(visible_points_from_lpc(lpc, geometry, frame_idx))
    target_points = [points for points in target_points if len(points) > 0]
    if not target_points:
        return None, []
    target_points_combined = np.concatenate(target_points, axis=0)

    scored = []
    for hist_idx, hist_points in frame_visible_points.items():
        iou = compute_points_iou(target_points_combined, hist_points, voxel_size)
        if iou >= iou_threshold:
            scored.append((hist_idx, iou))
    scored.sort(key=lambda item: item[1], reverse=True)

    selected = [idx for idx, _ in scored[:max_reference_frames]]
    if not selected:
        return None, []
    ref_latents = torch.stack([generated_latents[idx] for idx in selected], dim=1)
    return ref_latents.unsqueeze(0), selected


def visible_points_from_lpc(
    lpc: LatentPointCloud,
    geometry: VideoGeometry,
    frame_idx: int,
) -> np.ndarray:
    device = lpc.points_world.device
    points_world = lpc.points_world[lpc.valid_mask.bool()]
    cam2world = torch.as_tensor(
        geometry.poses_c2w[frame_idx],
        device=device,
        dtype=torch.float32,
    )
    intrinsics = torch.as_tensor(
        scale_intrinsics_to_hw(
            geometry.intrinsics[frame_idx],
            geometry.frames.shape[1:3],
            lpc.latent_hw,
        ),
        device=device,
        dtype=torch.float32,
    )

    world2cam = torch.inverse(cam2world)
    points_cam = (points_world @ world2cam[:3, :3].T) + world2cam[:3, 3]
    z = points_cam[:, 2]
    u = points_cam[:, 0] * intrinsics[0, 0] / z + intrinsics[0, 2]
    v = points_cam[:, 1] * intrinsics[1, 1] / z + intrinsics[1, 2]
    height, width = lpc.latent_hw
    valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return points_world[valid].detach().cpu().numpy().astype(np.float32)


def compute_points_iou(
    points_a: np.ndarray,
    points_b: np.ndarray,
    voxel_size: float,
) -> float:
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0
    vox_a = np.floor(points_a / voxel_size).astype(np.int32)
    vox_b = np.floor(points_b / voxel_size).astype(np.int32)
    set_a = set(map(tuple, vox_a.tolist()))
    set_b = set(map(tuple, vox_b.tolist()))
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def update_frame_visibility(
    *,
    lpc: LatentPointCloud,
    geometry: VideoGeometry,
    pose_indices: list[int],
    frame_visible_points: dict[int, np.ndarray],
    start_output_latent: int,
) -> None:
    for offset, pose_idx in enumerate(pose_indices):
        frame_visible_points[start_output_latent + offset] = visible_points_from_lpc(
            lpc,
            geometry,
            pose_idx,
        )


def build_vace_context_96(
    *,
    pipe: WanVideoPipeline,
    target_scene: Tensor,
    preceding_scene: Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
) -> list[Tensor]:
    if preceding_scene is not None:
        scene = torch.cat([target_scene, preceding_scene], dim=1)
    else:
        scene = target_scene
    if scene.shape[0] != 48:
        raise ValueError(f"Expected 48-channel scene latent, got {scene.shape[0]}.")

    score_latent = build_dummy_score_latent(
        pipe=pipe,
        scene=scene,
        dtype=dtype,
        device=device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    vace = torch.cat([scene.to(device=device, dtype=dtype), score_latent], dim=0)
    assert vace.shape[0] == 96, f"VACE context must have 96 channels, got {vace.shape}"
    return [vace]


def build_dummy_score_latent(
    *,
    pipe: WanVideoPipeline,
    scene: Tensor,
    dtype: torch.dtype,
    device: torch.device,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
) -> Tensor:
    _, latent_frames, latent_h, latent_w = scene.shape
    pixel_frames = (latent_frames - 1) * 4 + 1
    pixel_h = latent_h * pipe.vae.upsampling_factor
    pixel_w = latent_w * pipe.vae.upsampling_factor
    vae_dtype = next(pipe.vae.parameters()).dtype
    score_video = torch.ones(
        (3, pixel_frames, pixel_h, pixel_w),
        device=device,
        dtype=vae_dtype,
    )
    score_latent = pipe.vae.encode(
        [score_video],
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    ).to(device=device, dtype=dtype)[0]
    expected = (48, latent_frames, latent_h, latent_w)
    if tuple(score_latent.shape) != expected:
        raise ValueError(
            "Dummy control score latent shape mismatch: "
            f"expected {expected}, got {tuple(score_latent.shape)}."
        )
    return score_latent


def infer_mapanything_depths(
    *,
    images: np.ndarray,
    geometry: VideoGeometry,
    pose_indices: list[int],
    model_id: str,
    device: torch.device,
    model_cache: MiragePipeline,
) -> list[dict[str, np.ndarray]]:
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    if model_cache.mapanything_model is None:
        model_cache.mapanything_model = MapAnything.from_pretrained(model_id).to(device)
        model_cache.mapanything_model.eval()

    image_hw = images.shape[1:3]
    views = []
    for image, pose_idx in zip(images, pose_indices, strict=True):
        views.append(
            {
                "img": image,
                "intrinsics": scale_intrinsics_to_hw(
                    geometry.intrinsics[pose_idx],
                    geometry.frames.shape[1:3],
                    image_hw,
                ).astype(np.float32),
                "camera_poses": geometry.poses_c2w[pose_idx].astype(np.float32),
                "is_metric_scale": torch.tensor([True]),
            }
        )

    with torch.no_grad():
        predictions = model_cache.mapanything_model.infer(
            preprocess_inputs(views),
            memory_efficient_inference=True,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )

    output = []
    for pred in predictions:
        output.append(
            {
                "depth": pred["depth_z"]
                .detach()
                .cpu()
                .numpy()
                .squeeze(0)
                .squeeze(-1)
                .astype(np.float32),
            }
        )
    return output


def scale_intrinsics_to_hw(
    intrinsics: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    scaled = intrinsics.copy().astype(np.float32)
    scaled[0, 0] *= target_w / source_w
    scaled[0, 2] *= target_w / source_w
    scaled[1, 1] *= target_h / source_h
    scaled[1, 2] *= target_h / source_h
    return scaled


def select_latent_aligned_frames(video: np.ndarray, temporal_stride: int) -> np.ndarray:
    return video[::temporal_stride]


def write_iteration_video(
    path: Path,
    video: np.ndarray,
    iter_idx: int,
    fps: int,
) -> None:
    frames = video if iter_idx == 0 else video[1:]
    write_mp4(path, frames, fps)


def write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    video = frames
    if video.dtype != np.uint8:
        video = np.clip(video, 0, 255).astype(np.uint8)

    writer = None
    try:
        writer = imageio.get_writer(
            str(path),
            format="FFMPEG",
            mode="I",
            fps=float(fps),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
            ffmpeg_log_level="error",
        )
        for frame in video:
            writer.append_data(frame)
    finally:
        if writer is not None:
            writer.close()


def load_lora_checkpoint(
    pipe: WanVideoPipeline,
    path: Path,
    *,
    alpha: float,
) -> None:
    state = load_state_dict(str(path), torch_dtype=pipe.torch_dtype, device=pipe.device)
    state = _normalize_lora_state_dict(state)
    pipe.load_lora(pipe.dit, lora_state_dict=state, alpha=alpha)


def _normalize_lora_state_dict(state: dict[str, Any]) -> dict[str, Tensor]:
    normalized_state: dict[str, Tensor] = {}
    source_keys: dict[str, str] = {}

    for key, value in state.items():
        if not torch.is_tensor(value) or "lora_" not in key:
            continue

        normalized_key = _normalize_lora_checkpoint_key(key)
        if normalized_key in normalized_state:
            raise ValueError(
                "LoRA checkpoint contains duplicate tensors after normalization: "
                f"'{source_keys[normalized_key]}' and '{key}' both map to "
                f"'{normalized_key}'."
            )

        normalized_state[normalized_key] = value
        source_keys[normalized_key] = key

    if not normalized_state:
        raise ValueError("No LoRA tensors found in LoRA checkpoint.")

    return normalized_state


def _normalize_lora_checkpoint_key(key: str) -> str:
    normalized = strip_prefixes(
        key,
        (
            "module.",
            "model.",
            "pipe.dit.",
            "dit.",
            "base_model.model.",
            "base_model.",
        ),
    )
    normalized = normalized.replace(".lora_A.default.weight", ".lora_A.weight")
    normalized = normalized.replace(".lora_B.default.weight", ".lora_B.weight")
    return normalized
