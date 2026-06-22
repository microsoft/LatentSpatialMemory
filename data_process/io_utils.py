from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_process.naming import get_sample_naming
from data_process.types import VideoGeometry


def save_sample_npz(path: str | Path, sample: dict, naming: str = "figure") -> None:
    """Save training sample to NPZ format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = get_sample_naming(naming)

    payload = {
        names.preceding_rgb: sample["P_rgb"],
        names.target_rgb: sample["T_rgb"],
        names.preceding_poses_c2w: sample.get("P_poses_c2w"),
        names.target_poses_c2w: sample.get("T_poses_c2w"),
        names.preceding_intrinsics: sample.get("P_intrinsics"),
        names.target_intrinsics: sample.get("T_intrinsics"),
        names.scene_xyz: sample["scene_xyz"],
        names.preceding_scene_proj: sample["proj_P"],
        names.target_scene_proj: sample["proj_T"],
        names.reference_rgb: sample["R_rgb"],
    }
    if "scene_rgb" in sample and sample["scene_rgb"] is not None:
        payload[names.scene_rgb] = sample["scene_rgb"]

    payload = {k: v for k, v in payload.items() if v is not None}
    np.savez_compressed(path, **payload)

    meta_path = path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        meta = dict(sample.get("meta", {}))
        meta["naming"] = naming
        json.dump(meta, f, indent=2)


# Backward compatibility alias
save_episode_npz = save_sample_npz


def save_video_geometry(path: str | Path, geometry: VideoGeometry) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=geometry.frames,
        depths=geometry.depths,
        intrinsics=geometry.intrinsics,
        poses_c2w=geometry.poses_c2w,
        masks=geometry.masks if geometry.masks is not None else np.array([]),
        frame_indices=geometry.frame_indices
        if geometry.frame_indices is not None
        else np.array([]),
        original_size=np.array(
            geometry.original_size if geometry.original_size else (-1, -1)
        ),
        processed_size=np.array(
            geometry.processed_size if geometry.processed_size else (-1, -1)
        ),
    )


def load_video_geometry(path: str | Path) -> VideoGeometry:
    data = np.load(path, allow_pickle=True)
    masks = data["masks"]
    if masks.size == 0:
        masks = None
    frame_indices = data["frame_indices"]
    if frame_indices.size == 0:
        frame_indices = None
    original_size = tuple(data["original_size"].tolist())
    if original_size == (-1, -1):
        original_size = None
    processed_size = tuple(data["processed_size"].tolist())
    if processed_size == (-1, -1):
        processed_size = None

    return VideoGeometry(
        frames=data["frames"],
        depths=data["depths"],
        intrinsics=data["intrinsics"],
        poses_c2w=data["poses_c2w"],
        masks=masks,
        frame_indices=frame_indices,
        original_size=original_size,
        processed_size=processed_size,
    )
