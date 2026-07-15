"""
All configuration in one place.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOG_LEVEL", "WARNING")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ============================================================================
# ALL CONFIG HERE - Edit these values directly
# ============================================================================


@dataclass
class DataConfig:
    # === Paths ===
    # video_dirs: list of RAW source video directories (will be sampled into clips)
    video_dirs: list[str] = field(
        default_factory=lambda: [
            "data/spatialvid-hq/SpatialVID/videos",
        ]
    )
    output_root: str = "data/train"

    # === Clip extraction (from raw videos) ===
    clip_num_frames: int = 81  # Number of frames per clip
    clip_target_fps: int = 16  # Target FPS for clip
    clip_target_width: int = 1280  # Target width
    clip_target_height: int = 704  # Target height

    # === General pipeline behavior ===
    max_videos: Optional[int] = 4
    shuffle_seed: Optional[int] = 41
    fps_override: Optional[float] = None  # If set, override clip FPS for downstream
    naming_style: str = "figure"  # "figure", "paper", or "legacy"
    skip_existing: bool = True
    quiet_nonzero: bool = True
    cleanup_interval: int = 50

    # === Video VAE encoding ===
    video_vae_model_path: str = "data/Wan-AI/Wan2.2-TI2V-5B"
    video_vae_checkpoint: str = "Wan2.2_VAE.pth"

    # === Sample building ===
    N_target: int = 33  # Number of target frames
    M_pre: int = 8  # Number of preceding (prefix) frames
    min_gap_for_candidates: int = 2
    K_ref_stride: int = 2  # Stride for reference frame candidates
    eps_iou: float = 0.04  # IOU threshold for reference selection
    max_refs: int = 8  # Maximum reference frames
    point_cloud_vae_model_path: str = "data/Wan-AI/Wan2.2-TI2V-5B"
    point_cloud_vae_checkpoint: str = "Wan2.2_VAE.pth"
    point_cloud_vae_dtype: str = "bf16"  # "fp32", "fp16", or "bf16"
    scene_voxel_size: float = 0.01  # Reference-frame IOU uses 10x automatically.
    num_samples: int = 1  # Number of training samples per video
    sample_random_seed: Optional[int] = None

    def print_config(self) -> None:
        print("=" * 60)
        print("Mirage Pipeline Configuration")
        print("=" * 60)
        print(f"Video dirs      : {self.video_dirs}")
        print(f"Output root     : {self.output_root}")
        print(f"Max videos      : {self.max_videos}")
        print(f"Skip existing   : {self.skip_existing}")
        print("--- Clip Extraction ---")
        print(f"  Num frames    : {self.clip_num_frames}")
        print(f"  Target FPS    : {self.clip_target_fps}")
        print(f"  Resolution    : {self.clip_target_width}x{self.clip_target_height}")
        print("--- Video VAE ---")
        print(f"  Model path    : {self.video_vae_model_path}")
        print(f"  Checkpoint    : {self.video_vae_checkpoint}")
        print("--- Sample ---")
        print(f"  N_target      : {self.N_target}")
        print(f"  M_pre         : {self.M_pre}")
        print(f"  eps_iou       : {self.eps_iou}")
        print(f"  max_refs      : {self.max_refs}")
        print(f"  point_vae_dty : {self.point_cloud_vae_dtype}")
        print(f"  voxel_size    : {self.scene_voxel_size}")
        print("=" * 60)


# ============================================================================
# Global config instance - import and use this directly
# ============================================================================

CONFIG = DataConfig()


# ============================================================================
# Helper: Load config from file (optional, for advanced use)
# ============================================================================


def load_config(path: str | Path | None = None) -> DataConfig:
    """
    Load config from JSON/YAML file, or return the global CONFIG.

    For most cases, just import and use CONFIG directly.
    """
    if path is None or str(path).strip().lower() in {"", "none", "null"}:
        return CONFIG

    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML required for YAML configs") from exc
        data = yaml.safe_load(path.read_text())
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError("Config must be a dictionary")

    # Create config from dict, using defaults for missing keys
    return DataConfig(**{k: v for k, v in data.items() if hasattr(DataConfig, k)})


@dataclass
class SampleConfig:
    """Training sample building config extracted from the global data config."""

    N_target: int = CONFIG.N_target
    M_pre: int = CONFIG.M_pre
    min_gap_for_candidates: int = CONFIG.min_gap_for_candidates
    K_ref_stride: int = CONFIG.K_ref_stride
    eps_iou: float = CONFIG.eps_iou
    max_refs: int = CONFIG.max_refs
    point_cloud_vae_model_path: str = CONFIG.point_cloud_vae_model_path
    point_cloud_vae_checkpoint: str = CONFIG.point_cloud_vae_checkpoint
    point_cloud_vae_dtype: str = CONFIG.point_cloud_vae_dtype
    scene_voxel_size: float = CONFIG.scene_voxel_size
    ref_iou_voxel_size: Optional[float] = None  # If None, uses scene_voxel_size * 10
    num_samples: int = CONFIG.num_samples
    random_seed: Optional[int] = CONFIG.sample_random_seed


# Backward compatibility alias
EpisodeConfig = SampleConfig


def get_sample_config() -> SampleConfig:
    """Get SampleConfig from global CONFIG."""
    return SampleConfig()


# Backward compatibility alias
get_episode_config = get_sample_config
