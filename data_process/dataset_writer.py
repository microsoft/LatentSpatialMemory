"""
数据集写入模块

将训练样本保存为文件:
- mp4视频: preceding_rgb, target_rgb, reference_rgb, projection等
- json: 样本元信息 (帧索引、参考帧IOU等)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from data_process.naming import get_sample_naming
from data_process.video_io import save_mp4
from data_process.viz import save_point_cloud_ply


def save_frames(frames: np.ndarray, out_dir: str | Path, ext: str = "png") -> None:
    """保存帧序列为图片"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        img = frame
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(out_dir / f"{idx:05d}.{ext}")


def save_video(frames: np.ndarray, out_path: str | Path, fps: float = 16.0) -> None:
    """保存帧序列为mp4视频"""
    save_mp4(frames, out_path, fps=fps)


def save_projection_rgb_frames(
    proj: np.ndarray, out_dir: str | Path, ext: str = "png"
) -> None:
    if proj.ndim != 4 or proj.shape[-1] != 3:
        raise ValueError("proj must be (N,H,W,3)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(proj):
        img = np.clip(frame, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(out_dir / f"{idx:05d}.{ext}")


def save_projection_rgb_video(
    proj: np.ndarray, out_path: str | Path, fps: float = 16.0
) -> None:
    if proj.ndim != 4 or proj.shape[-1] != 3:
        raise ValueError("proj must be (N,H,W,3)")
    save_video(proj, out_path, fps=fps)


def save_projection_latent(
    proj: np.ndarray,
    out_path: str | Path,
) -> None:
    """Save latent scene projection to a torch file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(proj).to(dtype=torch.bfloat16)
    torch.save({"latent": tensor}, out_path)


def save_pose_txt(poses_c2w: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    lines = [str(len(poses_c2w))]
    for idx, pose in enumerate(poses_c2w):
        vals = " ".join(f"{v:.6f}" for v in pose.reshape(-1).tolist())
        lines.append(f"{idx} {vals}")
    out_path.write_text("\n".join(lines))


def save_intrinsics_txt(intrinsics: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    lines = [str(len(intrinsics))]
    for idx, K in enumerate(intrinsics):
        vals = " ".join(f"{v:.6f}" for v in K.reshape(-1).tolist())
        lines.append(f"{idx} {vals}")
    out_path.write_text("\n".join(lines))


def save_training_sample(
    sample: dict,
    out_dir: str | Path,
    projection_channels: list[str],
    save_npz: bool = True,
    fps: float = 16.0,
    save_frame_dirs: bool = False,
    naming: str = "figure",
    name_prefix: str = "",
    videos_only: bool = False,
) -> None:
    """
    Save training sample outputs.

    Simplified output structure:
    - Videos: preceding_rgb, target_rgb, reference_rgb,
              preceding_scene_proj_rgb, target_scene_proj_rgb
    - sample.json: Contains all meta info (scene_idx, R_idx, R_iou, etc.)

    No longer saves: .npz, .ply, pose/intrinsics txt files (these can be
    reconstructed from geometry.npz + sample.json)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = get_sample_naming(naming)
    prefix = name_prefix
    meta = dict(sample.get("meta", {}))
    point_cloud_type = meta.get("point_cloud_type", "explicit")

    # Save videos
    save_video(sample["P_rgb"], out_dir / f"{prefix}{names.preceding_rgb}.mp4", fps=fps)
    save_video(sample["T_rgb"], out_dir / f"{prefix}{names.target_rgb}.mp4", fps=fps)
    if sample["R_rgb"].shape[0] > 0:
        save_video(
            sample["R_rgb"], out_dir / f"{prefix}{names.reference_rgb}.mp4", fps=fps
        )
    # Save projection RGB videos
    if point_cloud_type == "latent":
        save_projection_latent(
            sample["proj_P"],
            out_dir / f"{prefix}{names.preceding_scene_proj_rgb}.pt",
        )
        save_projection_latent(
            sample["proj_T"],
            out_dir / f"{prefix}{names.target_scene_proj_rgb}.pt",
        )
    else:
        channels = list(projection_channels)
        cursor = 0
        rgb_slice = None
        for name in channels:
            if name == "rgb":
                rgb_slice = slice(cursor, cursor + 3)
                cursor += 3
            else:
                cursor += 1

        if rgb_slice is not None:
            preceding_proj_rgb = sample["proj_P"][..., rgb_slice]
            target_proj_rgb = sample["proj_T"][..., rgb_slice]
            save_projection_rgb_video(
                preceding_proj_rgb,
                out_dir / f"{prefix}{names.preceding_scene_proj_rgb}.mp4",
                fps=fps,
            )
            save_projection_rgb_video(
                target_proj_rgb,
                out_dir / f"{prefix}{names.target_scene_proj_rgb}.mp4",
                fps=fps,
            )

    # Save sample.json (meta info for reconstruction)
    meta["naming"] = naming
    meta_path = out_dir / f"{prefix}sample.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# Backward compatibility alias
save_episode_dir = save_training_sample
