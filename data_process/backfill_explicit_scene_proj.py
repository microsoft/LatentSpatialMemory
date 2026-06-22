#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import multiprocessing as mp
import queue
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from data_process._0_0_0_root_assign import CONFIG
from data_process.distributed import shard_items
from data_process.point_cloud import build_scene_point_cloud
from data_process.projection import render_projection
from data_process.run_video_vae_encode import get_vae_wrapper
from data_process.types import VideoGeometry
from data_process.video_io import load_video_frames

REQUIRED_FILES = (
    "train_sample.json",
    "geometry.npz",
    "clip.mp4",
    "dynamic_masks.npy",
)
OUTPUT_PRECEDING = "train_preceding_scene_proj_rgb_explicit.pt"
OUTPUT_TARGET = "train_target_scene_proj_rgb_explicit.pt"


def _parse_visible_cuda_devices() -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return list(range(torch.cuda.device_count()))

    visible_tokens = [item.strip() for item in visible.split(",") if item.strip()]
    return list(range(len(visible_tokens)))


def _resolve_worker_devices(device: str) -> list[str]:
    normalized = device.strip()
    if normalized == "cpu":
        return ["cpu"]

    if normalized.startswith("cuda:"):
        return [normalized]

    if normalized != "cuda":
        raise ValueError(f"Unsupported device: {device}")

    visible_devices = _parse_visible_cuda_devices()
    if not visible_devices:
        raise RuntimeError("CUDA device requested but no visible GPUs were found.")
    return [f"cuda:{idx}" for idx in range(len(visible_devices))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill explicit scene projection latents for existing Spatia samples."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=CONFIG.output_root,
        help="Root directory that will be scanned recursively for sample folders.",
    )
    parser.add_argument(
        "--vae-model-path",
        type=str,
        default=CONFIG.video_vae_model_path,
        help="Wan model directory that contains Wan2.2 VAE weights.",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="Wan2.2_VAE.pth",
        help="VAE checkpoint filename. Defaults to Wan2.2_VAE.pth.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device used for VAE encoding.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip folders whose explicit outputs already exist. Existing files are never overwritten.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to all visible GPUs when --device=cuda.",
    )
    return parser.parse_args()


def discover_sample_dirs(input_root: Path) -> list[Path]:
    sample_dirs: set[Path] = set()
    for file_name in REQUIRED_FILES:
        for path in input_root.rglob(file_name):
            sample_dirs.add(path.parent)
    return sorted(sample_dirs)


def validate_required_files(sample_dir: Path) -> list[str]:
    missing = []
    for file_name in REQUIRED_FILES:
        if not (sample_dir / file_name).exists():
            missing.append(file_name)
    return missing


def load_sample_meta(sample_dir: Path) -> dict:
    meta_path = sample_dir / "train_sample.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_geometry(sample_dir: Path) -> tuple[VideoGeometry, tuple[int, int]]:
    geometry_path = sample_dir / "geometry.npz"
    clip_path = sample_dir / "clip.mp4"

    with np.load(geometry_path) as data:
        depths = np.asarray(data["depths"], dtype=np.float32)
        poses_c2w = np.asarray(data["poses_c2w"], dtype=np.float64)
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
        processed_size = tuple(int(x) for x in data["processed_size"])
        original_size = tuple(int(x) for x in data["original_size"])

    proc_h, proc_w = processed_size
    frames = np.asarray(load_video_frames(clip_path, target_size=(proc_w, proc_h)))
    assert len(frames) == len(depths), (
        f"Frame count {len(frames)} does not match depth count {len(depths)} for {sample_dir}"
    )
    geometry = VideoGeometry(
        frames=frames,
        depths=depths,
        intrinsics=intrinsics,
        poses_c2w=poses_c2w,
        masks=None,
        frame_indices=np.arange(len(frames), dtype=np.int32),
        original_size=original_size,
        processed_size=processed_size,
    )
    return geometry, original_size


def normalize_hw(size: tuple[int, int], fallback: tuple[int, int]) -> tuple[int, int]:
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        return fallback
    return int(size[0]), int(size[1])


def resolve_output_size(meta: dict, fallback: tuple[int, int]) -> tuple[int, int]:
    output_size = meta.get("output_size")
    if output_size is None:
        return fallback
    assert len(output_size) == 2, f"Invalid output_size: {output_size}"
    return int(output_size[0]), int(output_size[1])


def resize_mask_sequence(masks: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    target_h, target_w = target_hw
    resized = []
    for mask in masks:
        if mask.shape != (target_h, target_w):
            mask_u8 = mask.astype(np.uint8) * 255
            mask_u8 = cv2.resize(
                mask_u8,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = mask_u8 > 0
        resized.append(mask)
    return np.stack(resized, axis=0)


def load_dynamic_masks(sample_dir: Path, target_hw: tuple[int, int]) -> np.ndarray:
    masks = np.asarray(np.load(sample_dir / "dynamic_masks.npy")).astype(bool)
    return resize_mask_sequence(masks, target_hw)


def _scale_intrinsics(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 2] *= scale_y
    return K_scaled


def render_explicit_rgb_projection(
    geometry: VideoGeometry,
    scene_xyz: np.ndarray,
    scene_rgb: np.ndarray,
    frame_indices: Iterable[int],
    image_size: tuple[int, int],
) -> np.ndarray:
    proj_h, proj_w = image_size
    src_h, src_w = geometry.frames.shape[1:3]
    scale_x = proj_w / src_w
    scale_y = proj_h / src_h

    projections = []
    for frame_idx in frame_indices:
        K_scaled = _scale_intrinsics(geometry.intrinsics[frame_idx], scale_x, scale_y)
        projection = render_projection(
            points_world=scene_xyz,
            K=K_scaled,
            c2w=geometry.poses_c2w[frame_idx],
            image_size=(proj_h, proj_w),
            channels=["rgb"],
            colors=scene_rgb,
            fill_holes_kernel=0,
        )
        projections.append(projection.astype(np.uint8))
    return np.stack(projections, axis=0)


def encode_projection_latent(
    vae,
    frames: np.ndarray,
    device: str,
) -> torch.Tensor:
    tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 127.5 - 1.0
    tensor = tensor.to(device=device, dtype=torch.bfloat16)
    tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode_to_latent(tensor)[0]
    return latent.cpu().to(dtype=torch.bfloat16)


def save_latent(latent: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": latent}, out_path)


def process_sample(
    sample_dir: Path,
    vae,
    device: str,
    skip_existing: bool = True,
) -> tuple[str, str]:
    missing_files = validate_required_files(sample_dir)
    if missing_files:
        reason = f"missing required files: {', '.join(missing_files)}"
        return "skipped_missing", reason

    preceding_out = sample_dir / OUTPUT_PRECEDING
    target_out = sample_dir / OUTPUT_TARGET
    if skip_existing and preceding_out.exists() and target_out.exists():
        return "skipped_existing", "explicit outputs already exist"

    meta = load_sample_meta(sample_dir)
    assert "t0" in meta, f"Missing t0 in {sample_dir / 'train_sample.json'}"
    assert "P_idx" in meta, f"Missing P_idx in {sample_dir / 'train_sample.json'}"
    assert "T_idx" in meta, f"Missing T_idx in {sample_dir / 'train_sample.json'}"

    scene_idx = int(meta["t0"])
    preceding_indices = [int(idx) for idx in meta["P_idx"]]
    target_indices = [int(idx) for idx in meta["T_idx"]]

    geometry, original_size = load_geometry(sample_dir)
    original_size = normalize_hw(original_size, fallback=geometry.processed_size)
    output_size = resolve_output_size(meta, fallback=original_size)
    dynamic_masks = load_dynamic_masks(sample_dir, geometry.frames.shape[1:3])

    assert len(dynamic_masks) == len(geometry.frames), (
        f"dynamic mask count {len(dynamic_masks)} does not match frame count "
        f"{len(geometry.frames)} for {sample_dir}"
    )

    scene_xyz, scene_rgb = build_scene_point_cloud(
        depth=geometry.depths[scene_idx],
        K=geometry.intrinsics[scene_idx],
        c2w=geometry.poses_c2w[scene_idx],
        rgb=geometry.frames[scene_idx],
        valid_mask=None,
        dynamic_mask=dynamic_masks[scene_idx],
        voxel_size=CONFIG.scene_voxel_size,
    )
    assert scene_xyz.size > 0, f"Empty point cloud for {sample_dir}"
    assert scene_rgb is not None, f"Missing scene RGB for {sample_dir}"

    preceding_projection = render_explicit_rgb_projection(
        geometry=geometry,
        scene_xyz=scene_xyz,
        scene_rgb=scene_rgb,
        frame_indices=preceding_indices,
        image_size=output_size,
    )
    target_projection = render_explicit_rgb_projection(
        geometry=geometry,
        scene_xyz=scene_xyz,
        scene_rgb=scene_rgb,
        frame_indices=target_indices,
        image_size=output_size,
    )

    preceding_latent = encode_projection_latent(
        vae, preceding_projection, device=device
    )
    target_latent = encode_projection_latent(vae, target_projection, device=device)

    wrote_files: list[str] = []
    if not preceding_out.exists():
        save_latent(preceding_latent, preceding_out)
        wrote_files.append(preceding_out.name)
    if not target_out.exists():
        save_latent(target_latent, target_out)
        wrote_files.append(target_out.name)

    if not wrote_files:
        return "skipped_existing", "explicit outputs already exist"
    return "processed", f"wrote {', '.join(wrote_files)}"


def print_summary(
    input_root: Path,
    total_samples: int,
    status_counter: Counter,
    reason_counter: Counter,
) -> None:
    print("\n=== Explicit Scene Projection Backfill Summary ===")
    print(f"Input root: {input_root}")
    print(f"Discovered sample directories: {total_samples}")
    print(f"Processed successfully: {status_counter['processed']}")
    print(f"Skipped existing: {status_counter['skipped_existing']}")
    print(f"Skipped missing files: {status_counter['skipped_missing']}")
    print(f"Failed: {status_counter['failed']}")
    if reason_counter:
        print("Reason breakdown:")
        for reason, count in sorted(reason_counter.items()):
            print(f"  {reason}: {count}")


def _process_sample_dirs(
    sample_dirs: list[Path],
    *,
    worker_name: str,
    vae_model_path: str,
    vae_checkpoint: str,
    device: str,
    skip_existing: bool,
) -> tuple[Counter[str], Counter[str]]:
    if not sample_dirs:
        return Counter(), Counter()

    if device.startswith("cuda"):
        device_index = int(device.split(":", maxsplit=1)[1])
        torch.cuda.set_device(device_index)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.set_grad_enabled(False)

    print(
        f"[{worker_name}] Loading VAE on {device} for {len(sample_dirs)} samples",
        flush=True,
    )
    vae = get_vae_wrapper(vae_model_path, vae_checkpoint).to(
        device=device,
        dtype=torch.bfloat16,
    )

    status_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    for sample_dir in sample_dirs:
        try:
            status, message = process_sample(
                sample_dir,
                vae=vae,
                device=device,
                skip_existing=skip_existing,
            )
        except Exception as exc:
            status = "failed"
            message = str(exc)

        status_counter[status] += 1
        if status != "processed":
            reason_counter[message] += 1
        print(f"[{worker_name}][{status}] {sample_dir}: {message}", flush=True)

    del vae
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return status_counter, reason_counter


def _worker_main(
    rank: int,
    world_size: int,
    device: str,
    sample_dirs: list[Path],
    args: argparse.Namespace,
    result_queue,
) -> None:
    worker_name = f"worker-{rank}"
    shard = shard_items(sample_dirs, rank, world_size)
    try:
        status_counter, reason_counter = _process_sample_dirs(
            shard,
            worker_name=worker_name,
            vae_model_path=args.vae_model_path,
            vae_checkpoint=args.vae_checkpoint,
            device=device,
            skip_existing=args.skip_existing,
        )
        result_queue.put(
            {
                "rank": rank,
                "status_counter": dict(status_counter),
                "reason_counter": dict(reason_counter),
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "rank": rank,
                "status_counter": {"failed": len(shard)},
                "reason_counter": {f"worker crashed: {exc}": len(shard)},
            }
        )
        raise


def _run_parallel(
    sample_dirs: list[Path],
    args: argparse.Namespace,
    worker_devices: list[str],
) -> tuple[Counter[str], Counter[str]]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for rank, device in enumerate(worker_devices):
        process = ctx.Process(
            target=_worker_main,
            args=(rank, len(worker_devices), device, sample_dirs, args, result_queue),
        )
        process.start()
        processes.append(process)

    status_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    for process in processes:
        process.join()
    for _ in processes:
        try:
            payload = result_queue.get_nowait()
        except queue.Empty:
            break
        status_counter.update(payload["status_counter"])
        reason_counter.update(payload["reason_counter"])

    failed_processes = [
        f"{process.pid}:{process.exitcode}"
        for process in processes
        if process.exitcode != 0
    ]
    if failed_processes:
        raise RuntimeError(
            "Worker processes exited abnormally: " + ", ".join(failed_processes)
        )
    if sum(status_counter.values()) != len(sample_dirs):
        raise RuntimeError(
            "Did not receive complete worker results: "
            f"expected {len(sample_dirs)} samples, got {sum(status_counter.values())}"
        )

    return status_counter, reason_counter


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    sample_dirs = discover_sample_dirs(input_root)
    print(f"Discovered {len(sample_dirs)} sample directories under {input_root}")

    if not sample_dirs:
        print_summary(
            input_root=input_root,
            total_samples=0,
            status_counter=Counter(),
            reason_counter=Counter(),
        )
        return

    worker_devices = _resolve_worker_devices(args.device)
    if args.num_workers is not None:
        if args.num_workers <= 0:
            raise ValueError("--num-workers must be positive.")
        worker_devices = worker_devices[: args.num_workers]
    if not worker_devices:
        raise RuntimeError("No worker devices were selected.")

    if len(worker_devices) == 1:
        print(f"Running in single-worker mode on {worker_devices[0]}")
        status_counter, reason_counter = _process_sample_dirs(
            sample_dirs,
            worker_name="worker-0",
            vae_model_path=args.vae_model_path,
            vae_checkpoint=args.vae_checkpoint,
            device=worker_devices[0],
            skip_existing=args.skip_existing,
        )
    else:
        print(
            f"Running in multi-worker mode with {len(worker_devices)} GPUs: "
            f"{', '.join(worker_devices)}",
            flush=True,
        )
        status_counter, reason_counter = _run_parallel(
            sample_dirs=sample_dirs,
            args=args,
            worker_devices=worker_devices,
        )

    print_summary(
        input_root=input_root,
        total_samples=len(sample_dirs),
        status_counter=status_counter,
        reason_counter=reason_counter,
    )


if __name__ == "__main__":
    main()
