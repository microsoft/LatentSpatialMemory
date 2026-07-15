#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import torch
from tqdm.auto import tqdm

from data_process.data_config import CONFIG
from data_process.distributed import get_rank_info, shard_items
from mirage.wan2_2 import WanVAEWrapper


def _infer_vae_checkpoint(model_path: str) -> str:
    if "Wan2.2" in model_path or "5B" in model_path:
        return "Wan2.2_VAE.pth"
    return "Wan2.1_VAE.pth"


def get_vae_wrapper(model_path: str, vae_checkpoint: str) -> WanVAEWrapper:
    resolved_checkpoint = vae_checkpoint.strip() or _infer_vae_checkpoint(model_path)
    print(
        f"[VAE] Using mirage WanVAEWrapper: path={model_path}, checkpoint={resolved_checkpoint}",
        flush=True,
    )
    return WanVAEWrapper(
        wan_model_path=model_path,
        vae_checkpoint=resolved_checkpoint,
    )


def _parse_video_keys(raw: str) -> list[str]:
    if not raw:
        return []
    keys = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if item.lower().endswith(".mp4"):
            item = item[:-4]
        keys.append(item)
    return keys


def _list_video_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _load_video(path: Path) -> torch.Tensor:
    array = iio.imread(str(path), plugin="pyav")
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Unexpected shape for video {path}: {array.shape}")
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return tensor


def _preprocess_video(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    video = tensor.to(device=device, dtype=torch.float32)
    video = video.div_(255.0).mul_(2.0).sub_(1.0)
    return video.to(dtype=torch.bfloat16)


def _encode_video(model: WanVAEWrapper, video_tensor: torch.Tensor) -> torch.Tensor:
    """Encode video tensor to latent. Input: [1, C, T, H, W], Output: [C, T', H', W']."""
    with torch.no_grad():
        latent = model.encode_to_latent(video_tensor)[0]
    return latent.cpu().to(dtype=torch.bfloat16)


def _encode_frames_independently(
    model: WanVAEWrapper, video_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Encode each frame independently to avoid temporal correlation in 3D VAE.
    Input: [1, C, T, H, W], Output: [C, T', H', W'] where each frame is encoded separately.

    For reference frames that should not have temporal relationships.
    """
    # video_tensor shape: [1, C, T, H, W]
    num_frames = video_tensor.shape[2]
    latents = []

    with torch.no_grad():
        for t in range(num_frames):
            # Extract single frame: [1, C, 1, H, W]
            frame = video_tensor[:, :, t : t + 1, :, :]
            latent = model.encode(frame)[0]  # [C', 1, H', W']
            latents.append(latent)

    # Concatenate along the latent time dimension: [C', T', H', W']
    combined = torch.cat(latents, dim=1)
    return combined.cpu().to(dtype=torch.bfloat16)


def _log_message(
    message: str,
    rank: int,
    world_size: int,
    level: str = "INFO",
    quiet_nonzero: bool = False,
) -> None:
    if quiet_nonzero and rank != 0 and level != "ERROR":
        return
    tag = f"[VAE][{level}][R{rank}/{world_size}]"
    print(f"{tag} {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode video latents with Wan VAE.")
    parser.add_argument("--input-root", type=str, default=CONFIG.output_root)
    parser.add_argument("--video-keys", type=str, required=True)
    parser.add_argument(
        "--vae-model-path",
        type=str,
        default=CONFIG.video_vae_model_path,
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default=CONFIG.video_vae_checkpoint,
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--quiet-nonzero", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_keys = _parse_video_keys(args.video_keys)
    if not video_keys:
        raise ValueError("video_keys is empty after parsing.")

    if not torch.cuda.is_available():
        raise RuntimeError("VAE encoding requires a CUDA-capable GPU.")

    rank, world_size, local_rank = get_rank_info()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    _log_message(
        "Loading VAE model...", rank, world_size, quiet_nonzero=args.quiet_nonzero
    )
    model = get_vae_wrapper(args.vae_model_path, args.vae_checkpoint).to(
        device=device,
        dtype=torch.bfloat16,
    )

    root = Path(args.input_root)
    video_dirs = _list_video_dirs(root)
    if args.max_videos is not None:
        video_dirs = video_dirs[: args.max_videos]
    video_dirs = shard_items(video_dirs, rank, world_size)

    pbar = tqdm(total=len(video_dirs), disable=rank != 0, desc="vae-encode")
    for video_dir in video_dirs:
        video_id = video_dir.name
        if rank == 0:
            pbar.set_description(f"vae-encode {video_id}")

        for key in video_keys:
            video_path = video_dir / f"{key}.mp4"
            out_path = video_dir / f"{key}.pt"
            if out_path.exists() and out_path.stat().st_size > 0:
                if args.skip_existing or "scene_proj" in key:
                    _log_message(
                        f"Skip {out_path} (exists)",
                        rank,
                        world_size,
                        level="SKIP",
                        quiet_nonzero=args.quiet_nonzero,
                    )
                    continue
            if not video_path.exists():
                # For reference videos, missing is expected (not all samples have references)
                if "reference" in key:
                    _log_message(
                        f"Skip {video_path} (not found)",
                        rank,
                        world_size,
                        level="SKIP",
                        quiet_nonzero=args.quiet_nonzero,
                    )
                else:
                    _log_message(
                        f"Missing {video_path}",
                        rank,
                        world_size,
                        level="WARN",
                        quiet_nonzero=args.quiet_nonzero,
                    )
                continue

            raw_video = None
            video_tensor = None
            try:
                raw_video = _load_video(video_path)
                video_tensor = _preprocess_video(raw_video, device)

                # For reference videos, encode each frame independently to avoid temporal correlation
                if "reference" in key:
                    latent = _encode_frames_independently(model, video_tensor)
                    _log_message(
                        f"Wrote {out_path} (frame-by-frame)",
                        rank,
                        world_size,
                        quiet_nonzero=args.quiet_nonzero,
                    )
                else:
                    latent = _encode_video(model, video_tensor)
                    _log_message(
                        f"Wrote {out_path}",
                        rank,
                        world_size,
                        quiet_nonzero=args.quiet_nonzero,
                    )

                torch.save({"latent": latent}, out_path)
            except Exception as exc:
                _log_message(
                    f"Failed {video_path}: {exc}",
                    rank,
                    world_size,
                    level="ERROR",
                    quiet_nonzero=args.quiet_nonzero,
                )
            finally:
                del raw_video, video_tensor

        if rank == 0:
            pbar.update(1)

    if rank == 0:
        pbar.close()

    _log_message("Done", rank, world_size, quiet_nonzero=args.quiet_nonzero)


if __name__ == "__main__":
    main()
