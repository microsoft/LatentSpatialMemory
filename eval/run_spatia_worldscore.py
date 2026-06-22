from __future__ import annotations

import os
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from worldscore.benchmark.helpers import GetHelpers
from worldscore.benchmark.utils.utils import check_model, empty_cache
from worldscore.common.utils import print_banner

from latent_mem.configs.inference_config import (
    InferenceConfig,
    InferenceGenerationConfig,
    LatentMemPipelineConfig,
)
from latent_mem.configs.model_config import Wan2_2_5BConfig
from latent_mem.configs.training_config import LoRAConfig
from latent_mem.inference.pipeline_latent_mem import LatentMemPipeline
from latent_mem.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from latent_mem.wrapper.wan.base import WanCLIPEncoder, WanTextEncoder, WanVAEWrapper
from latent_mem.wrapper.wan.bidirectonal_vace import BidirectionalWanWrapperVACE
from latent_mem.wrapper.wan.checkpoint_loader import load_generator_checkpoint

MODEL_NAME = "spatia"
WORLD_SCORE_ROOT = Path(__file__).resolve().parent.parent / "world-score-spatia"


@dataclass
class ModelBundle:
    """Reusable inference objects for WorldScore generation."""

    pipeline: LatentMemPipeline
    device: str
    num_frames: int
    resolution: tuple[int, int]
    focal_length: float


def build_parser() -> ArgumentParser:
    """Build the CLI for the hard-coded WorldScore runner."""
    parser = ArgumentParser(
        description="Run a hard-coded WorldScore generation pipeline for Spatia."
    )
    parser.add_argument(
        "--worldscore_json_file",
        type=str,
        default="",
        help="Optional WorldScore JSON file name under the selected visual movement.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate at most one assigned sample per rank.",
    )
    return parser


def load_config() -> Any:
    """Load the merged WorldScore config for Spatia."""
    base_config = OmegaConf.load(WORLD_SCORE_ROOT / "config/base_config.yaml")
    model_config = OmegaConf.load(WORLD_SCORE_ROOT / "config/model_configs/spatia.yaml")
    return OmegaConf.merge(base_config, model_config)


def _build_inference_config(config: Any) -> InferenceConfig:
    """Create the latent-mem inference config used by this script."""
    num_frames = int(config["frames"])
    fps = int(config["fps"])
    return InferenceConfig(
        pipeline="latent",
        num_frames=num_frames,
        generation=InferenceGenerationConfig(fps=fps),
        backbone=Wan2_2_5BConfig(),
    )


def _load_checkpoint_weights(
    generator: torch.nn.Module,
    inference_config: InferenceConfig,
) -> None:
    """Load the trained Spatia checkpoint into the generator."""
    if inference_config.generation.use_ema:
        raise ValueError("EMA checkpoints are no longer supported.")

    ckpt_file = Path(inference_config.generator_ckpt_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Generator checkpoint not found: {ckpt_file}")

    print(f"Loading checkpoint: {ckpt_file}")
    load_result = load_generator_checkpoint(
        generator,
        ckpt_file,
    )
    print(
        f"Loaded '{load_result.source_key}' weights: "
        f"{load_result.num_lora_tensors} LoRA tensors, "
        f"{load_result.num_vace_tensors} VACE tensors"
    )
    torch.cuda.empty_cache()


def load_model(config: Any, device: str) -> ModelBundle:
    """Load the latent-mem pipeline and its reusable dependencies."""
    inference_config = _build_inference_config(config)

    set_seed(inference_config.seed)
    generator_config = inference_config.backbone

    print("Initializing generator with VACE ControlNet...")
    generator = BidirectionalWanWrapperVACE(generator_config)
    print("Base model loaded")

    if generator_config.use_lora:
        print("Applying LoRA adapters...")
        lora_cfg = LoRAConfig()
        lora_config = LoraConfig(**asdict(lora_cfg))
        lora_config.exclude_modules = r".*(?:vace_blocks|vace_patch_embedding).*"
        generator = get_peft_model(generator, lora_config)
        for _, peft_config in generator.peft_config.items():
            peft_config.inference_mode = True
        print(f"LoRA applied with rank={lora_cfg.r}")

    _load_checkpoint_weights(generator, inference_config)

    generator.to(device=device, dtype=torch.bfloat16)
    generator.eval()
    print("Generator moved to device")

    print("Loading text encoder...")
    text_encoder = WanTextEncoder(model_name=generator_config.wan_model_path)
    text_encoder.text_encoder.to(device=device, dtype=torch.bfloat16)
    print("Text encoder loaded")

    print("Loading VAE...")
    vae = WanVAEWrapper(
        wan_model_path=generator_config.wan_model_path,
        vae_checkpoint=generator_config.vae_checkpoint,
    ).to(device=device, dtype=torch.bfloat16)
    print("VAE loaded")

    clip_encoder = None
    if generator_config.model_type == "i2v":
        print("Loading CLIP encoder...")
        clip_encoder = WanCLIPEncoder(model_name=generator_config.wan_model_path)
        clip_encoder.to(device=device, dtype=torch.bfloat16)
        print("CLIP encoder loaded")

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=inference_config.num_train_timestep,
        shift=1,
        use_dynamic_shifting=False,
    )
    pipeline_config = LatentMemPipelineConfig()
    pipeline = LatentMemPipeline(
        config=pipeline_config,
        generator=generator,
        vae=vae,
        text_encoder=text_encoder,
        clip_encoder=clip_encoder,
        scheduler=scheduler,
    )

    return ModelBundle(
        pipeline=pipeline,
        device=device,
        num_frames=int(config["frames"]),
        resolution=tuple(config["resolution"]),
        focal_length=float(config.get("focal_length", 500)),
    )


def _build_intrinsics(
    num_frames: int,
    resolution: tuple[int, int],
    focal_length: float,
) -> np.ndarray:
    """Construct pinhole intrinsics for each frame."""
    width, height = resolution
    intrinsics = np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.repeat(intrinsics[None, ...], num_frames, axis=0)


def generate(
    model: ModelBundle,
    conditioning_image: str,
    inpainting_prompt_list: list[str],
    cameras: Any,
    cameras_interp: Any,
    helper: Any,
    config: Any,
) -> Any:
    """
    Run model-specific generation and return frames in a WorldScore-compatible format.

    Expected return type:
    - list[PIL.Image.Image], or
    - torch.Tensor shaped [N, 3, H, W] with values in [0, 1]
    """
    poses_c2w = cameras_interp
    if isinstance(poses_c2w, torch.Tensor):
        poses_c2w = poses_c2w.detach().cpu().numpy()
    poses_c2w = np.asarray(poses_c2w, dtype=np.float32)

    if poses_c2w.shape[0] < model.num_frames:
        raise ValueError(
            f"WorldScore provided {poses_c2w.shape[0]} poses, expected at least {model.num_frames}"
        )

    poses_c2w_raw = poses_c2w
    # WorldScore stores Blender-style camera bases; latent-mem projection expects
    # OpenCV-style c2w with +Z in front of the camera.
    worldscore_to_infer = np.diag([-1.0, 1.0, -1.0, 1.0]).astype(np.float32)
    poses_c2w = poses_c2w @ worldscore_to_infer
    if bool(config.get("dry_run", False)):
        print("Pose convention conversion:")
        print(f"  raw R diag: {np.diag(poses_c2w_raw[0, :3, :3]).tolist()}")
        print(f"  new R diag: {np.diag(poses_c2w[0, :3, :3]).tolist()}")
        print(f"  raw t: {poses_c2w_raw[0, :3, 3].tolist()}")
        print(f"  new t: {poses_c2w[0, :3, 3].tolist()}")

    poses_c2w = poses_c2w[: model.num_frames]
    intrinsics = _build_intrinsics(
        num_frames=model.num_frames,
        resolution=model.resolution,
        focal_length=model.focal_length,
    )

    sample_dir = Path(helper.path)
    pointcloud_dir = sample_dir / "pointclouds"
    video_dir = sample_dir / "videos"
    pointcloud_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    prompt = inpainting_prompt_list[0] if inpainting_prompt_list else ""
    frames = model.pipeline.generate(
        first_frame_path=conditioning_image,
        text_prompt=prompt,
        poses_list=poses_c2w,
        intrinsics_list=intrinsics,
        t0=0,
        video_dir=video_dir,
        pointcloud_dir=pointcloud_dir,
    )
    return frames


def main(argv: list[str] | None = None) -> None:
    """
    Entry point for the hard-coded WorldScore pipeline.

    Flow:
    1. Parse CLI args.
    2. Load benchmark/model config.
    3. Initialize the model environment and load the model.
    4. Loop over `videogen` splits (`static` and `dynamic`).
    5. For each sample, adapt inputs, run generation, and save outputs.
    """
    print_banner("SPATIA HARDCODED GENERATION")
    parser = build_parser()
    args = parser.parse_args(argv)

    if not check_model(MODEL_NAME):
        raise ValueError(f"Model not exists: {MODEL_NAME}")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    print(f"Rank {rank}/{world_size} using device {device}")

    # Stage A: initialize config and model once.
    config = load_config()
    config["dry_run"] = args.dry_run
    model = load_model(config, device)
    generated_count = 0

    # Stage B: iterate through the two videogen benchmark splits.
    for visual_movement in ["static", "dynamic"]:
        # WorldScore API:
        # - prepares the dataloader for this split
        # - binds the model adapter through helper.adapt(...)
        dataloader, helper = GetHelpers(
            MODEL_NAME,
            visual_movement,
            args.worldscore_json_file,
        )

        for idx, data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            # Stage C: convert one benchmark sample into model-ready inputs.
            # helper.adapt(...) is provided by WorldScore's adapter mechanism.
            conditioning_image, inpainting_prompt_list, cameras, cameras_interp = (
                helper.adapt(data)
            )

            # Stage D: run model-specific generation.
            # This block is intentionally left blank for you to implement.
            frames = generate(
                model=model,
                conditioning_image=conditioning_image,
                inpainting_prompt_list=inpainting_prompt_list,
                cameras=cameras,
                cameras_interp=cameras_interp,
                helper=helper,
                config=config,
            )

            # Stage E: rely on WorldScore helper API for standard output layout.
            helper.save(frames)
            empty_cache()
            generated_count += 1

            if args.dry_run and generated_count >= 1:
                print(f"Rank {rank}: dry run complete after 1 generated sample")
                return


if __name__ == "__main__":
    main()
