"""
Spatia pipeline central configuration.

All configuration in ONE place. Edit this file to control Spatia data processing.
"""

from __future__ import annotations

import os

# ============================================================================
# Environment setup - MUST be before any other imports
# ============================================================================
os.environ["SAM3_TQDM_DISABLE"] = "1"  # Legacy compatibility for old SAM3 tooling
os.environ.setdefault("LOG_LEVEL", "WARNING")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ============================================================================
# ALL CONFIG HERE - Edit these values directly
# ============================================================================


@dataclass
class SpatiaConfig:
    """Single unified config for the entire Spatia pipeline."""

    # === Paths ===
    # video_dirs: list of RAW source video directories (will be sampled into clips)
    video_dirs: list[str] = field(
        default_factory=lambda: [
            "data/spatialvid-hq/SpatialVID/videos",
        ]
    )
    output_root: str = "data/Spatia/demo"

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
    save_raw_frames: bool = True
    save_frame_dirs: bool = False
    skip_existing: bool = True
    videos_only: bool = False
    quiet_nonzero: bool = True
    cleanup_interval: int = 50

    # === Qwen (entity extraction via DashScope API) ===
    disable_qwen: bool = False
    qwen_model_path: str = (
        "dashscope/qwen3-vl-flash"  # Legacy local path, used in video caption.sh
    )
    qwen_api_model: str = "dashscope/qwen3-vl-flash"
    qwen_prompt: Optional[str] = None
    max_prompts: int = 3
    qwen_batch_size: int = 8
    qwen_video_fps: float = 2.0
    qwen_video_min_frames: int = 4
    qwen_video_max_frames: int = 10

    # === Legacy SAM3 settings (not used by current run_spatia_pipeline) ===
    disable_sam3: bool = False
    sam3_frame_index: int = 0
    sam3_propagation_direction: str = "forward"
    sam3_start_frame_index: Optional[int] = None
    sam3_max_frame_num_to_track: Optional[int] = None
    sam3_mask_dilate: int = 10
    sam3_score_threshold: float = 0.1
    sam3_new_det_thresh: float = 0.3
    sam3_checkpoint_path: str = "data/facebook/sam3/sam3.pt"
    sam3_multi_prompt: bool = False
    sam3_resize_input_to: int = 560

    # === MapAnything (geometry estimation) ===
    map_anything_model_id: str = "data/facebook/map-anything"
    map_anything_device: str = "cuda"
    map_anything_resize_mode: str = "fixed_mapping"
    map_anything_size: Optional[int] = None
    map_anything_resolution_set: int = 518
    map_anything_memory_efficient: bool = True
    map_anything_use_amp: bool = True
    map_anything_amp_dtype: str = "fp16"  # "fp16", "bf16", or "fp32"
    map_anything_apply_mask: bool = True
    map_anything_mask_edges: bool = True
    map_anything_apply_confidence_mask: bool = False
    map_anything_confidence_percentile: int = 10
    map_anything_chunk_size: Optional[int] = None
    map_anything_chunk_overlap: int = 0

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
    point_cloud_type: str = "latent"  # "explicit" or "latent"
    point_cloud_vae_model_path: str = "data/Wan-AI/Wan2.2-TI2V-5B"
    point_cloud_vae_checkpoint: str = "Wan2.2_VAE.pth"
    point_cloud_vae_dtype: str = "bf16"  # "fp32", "fp16", or "bf16"
    scene_voxel_size: float = (
        0.01  # Voxel size for scene point cloud (IOU uses 10x automatically)
    )
    projection_channels: list[str] = field(default_factory=lambda: ["latent"])
    projection_fill_kernel: int = 0
    num_samples: int = 1  # Number of training samples per video
    sample_random_seed: Optional[int] = None

    def print_config(self) -> None:
        print("=" * 60)
        print("Spatia Pipeline Configuration")
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
        print(f"  point_cloud   : {self.point_cloud_type}")
        print(f"  point_vae_dty : {self.point_cloud_vae_dtype}")
        print(f"  voxel_size    : {self.scene_voxel_size}")
        print("=" * 60)


# ============================================================================
# Global config instance - import and use this directly
# ============================================================================

CONFIG = SpatiaConfig()


# ============================================================================
# Helper: Load config from file (optional, for advanced use)
# ============================================================================


def load_config(path: str | Path | None = None) -> SpatiaConfig:
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
    return SpatiaConfig(**{k: v for k, v in data.items() if hasattr(SpatiaConfig, k)})


# ============================================================================
# Backward compatibility - these classes are used by other modules
# ============================================================================


@dataclass
class SampleConfig:
    """Training sample building config (extracted from SpatiaConfig for module compatibility)."""

    N_target: int = CONFIG.N_target
    M_pre: int = CONFIG.M_pre
    min_gap_for_candidates: int = CONFIG.min_gap_for_candidates
    K_ref_stride: int = CONFIG.K_ref_stride
    eps_iou: float = CONFIG.eps_iou
    max_refs: int = CONFIG.max_refs
    point_cloud_type: str = CONFIG.point_cloud_type
    point_cloud_vae_model_path: str = CONFIG.point_cloud_vae_model_path
    point_cloud_vae_checkpoint: str = CONFIG.point_cloud_vae_checkpoint
    point_cloud_vae_dtype: str = CONFIG.point_cloud_vae_dtype
    scene_voxel_size: float = CONFIG.scene_voxel_size
    ref_iou_voxel_size: Optional[float] = None  # If None, uses scene_voxel_size * 10
    projection_channels: list[str] = field(
        default_factory=lambda: list(CONFIG.projection_channels)
    )
    projection_fill_kernel: int = CONFIG.projection_fill_kernel
    num_samples: int = CONFIG.num_samples
    random_seed: Optional[int] = CONFIG.sample_random_seed


# Backward compatibility alias
EpisodeConfig = SampleConfig


@dataclass
class MapAnythingConfig:
    """MapAnything config (extracted from SpatiaConfig for module compatibility)."""

    model_id: str = CONFIG.map_anything_model_id
    device: str = CONFIG.map_anything_device
    resize_mode: str = CONFIG.map_anything_resize_mode
    size: Optional[int] = CONFIG.map_anything_size
    resolution_set: int = CONFIG.map_anything_resolution_set
    memory_efficient_inference: bool = CONFIG.map_anything_memory_efficient
    use_amp: bool = CONFIG.map_anything_use_amp
    amp_dtype: str = CONFIG.map_anything_amp_dtype
    apply_mask: bool = CONFIG.map_anything_apply_mask
    mask_edges: bool = CONFIG.map_anything_mask_edges
    apply_confidence_mask: bool = CONFIG.map_anything_apply_confidence_mask
    confidence_percentile: int = CONFIG.map_anything_confidence_percentile
    chunk_size: Optional[int] = CONFIG.map_anything_chunk_size
    chunk_overlap: int = CONFIG.map_anything_chunk_overlap


def get_sample_config() -> SampleConfig:
    """Get SampleConfig from global CONFIG."""
    return SampleConfig()


# Backward compatibility alias
get_episode_config = get_sample_config


def get_map_anything_config() -> MapAnythingConfig:
    """Get MapAnythingConfig from global CONFIG."""
    return MapAnythingConfig()
