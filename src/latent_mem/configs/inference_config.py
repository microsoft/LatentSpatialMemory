from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from omegaconf import MISSING

from latent_mem.configs.model_config import BackboneConfig


@dataclass
class InferenceGenerationConfig:
    fps: int = 16
    infer_steps: int = 20
    guidance_scale: Optional[float] = None
    no_cfg: bool = True
    use_ema: bool = False
    single_pass: bool = False


@dataclass
class InferenceSceneConfig:
    generate_scene_proj: bool = False


@dataclass
class SpatiaPipelineConfig:
    scene_data_dir: Path = None
    voxel_size: float = 0.01
    qwen_model_path: Path = Path(
        "hf_cache/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203/"
    )
    sam3_model_path: Path = Path("hf_cache/facebook--sam3/sam3.pt")
    # Model / VAE architecture constants
    image_or_video_shape: List[int] = field(default_factory=lambda: [1, 9, 16, 60, 104])
    vae_stride: List[int] = field(default_factory=lambda: [4, 8, 8])
    # Sampling / scheduler settings
    timestep_shift: float = 5.0
    guidance_scale: float = 7.0
    negative_prompt: str = (
        "过曝，静态，细节模糊不清，字幕，静止，最差质量，低质量，JPEG压缩残留，"
        "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
        "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
        "背景人很多，倒着走"
    )
    num_train_timestep: int = 1000


@dataclass
class LatentMemPipelineConfig:
    output_frames: int = 33
    infer_steps: int = 20
    use_cfg: bool = False
    guidance_scale: float = 1.0
    fps: int = 16
    preceding_pixel_frames: int = 8
    single_pass: bool = False


@dataclass
class InferenceConfig:
    defaults: list = field(
        default_factory=lambda: [
            "_self_",
            {"backbone": "5b"},
        ]
    )
    seed: int = 42
    custom_poses: bool = False
    custom_poses_filename: str = "custom_poses.npz"
    closed_loop: bool = False
    start_frame: Optional[int] = None
    num_frames: int = 33
    input_dir: Path = Path("data/example_imgs/00000020")
    output_dir: Path = Path("outputs/")
    generator_ckpt_path: Path = Path("data/model.pt")
    pipeline: str = "latent"  # "spatia latent"
    num_train_timestep: int = 1000

    generation: InferenceGenerationConfig = field(
        default_factory=InferenceGenerationConfig
    )
    scene: InferenceSceneConfig = field(default_factory=InferenceSceneConfig)
    backbone: BackboneConfig = MISSING
