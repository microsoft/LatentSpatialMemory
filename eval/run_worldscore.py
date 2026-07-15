#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
WORLD_SCORE_ROOT = REPO_ROOT.parent / "world-score-mirage"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(WORLD_SCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORLD_SCORE_ROOT))

os.environ.setdefault("WORLDSCORE_PATH", str(WORLD_SCORE_ROOT))

import numpy as np
import torch
from scripts.infer import (  # noqa: E402
    load_pipeline_from_args,
    validate_args,
)
from omegaconf import OmegaConf
from PIL import Image
from worldscore.benchmark.helpers import GetHelpers
from worldscore.benchmark.utils.utils import check_model, empty_cache
from worldscore.common.utils import print_banner

from mirage.inference.mirage_pipeline import (  # noqa: E402
    VideoGeometry,
    infer_mapanything_depths,
)

MODEL_NAME = "mirage"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WorldScore with Mirage and latent memory."
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
    parser.add_argument(
        "--model-config",
        action="append",
        required=True,
        help=(
            "Repeatable DiffSynth ModelConfig. Accepts a local path, "
            "'model_id:origin_file_pattern', 'key=value,key=value', or JSON."
        ),
    )
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--vace-checkpoint", type=Path, default=None)
    parser.add_argument("--lora-checkpoint", type=Path, default=None)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--infer-steps", type=int, default=20)
    parser.add_argument("--torch-dtype", type=str, default="bf16")
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--timestep-shift", type=float, default=5.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--no-cfg", dest="no_cfg", action="store_true", default=True)
    parser.add_argument("--use-cfg", dest="no_cfg", action="store_false")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--max-reference-frames", type=int, default=4)
    parser.add_argument("--preceding-pixel-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--mapanything-model-id",
        type=str,
        default="facebook/map-anything",
    )
    parser.add_argument("--ref-iou-threshold", type=float, default=0.04)
    parser.add_argument("--ref-iou-voxel-size", type=float, default=0.1)
    parser.add_argument("--no-tiled", action="store_true")
    parser.add_argument("--tile-size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile-stride", type=int, nargs=2, default=(15, 26))
    return parser


def load_worldscore_config() -> Any:
    base_config = OmegaConf.load(WORLD_SCORE_ROOT / "config/base_config.yaml")
    model_config = OmegaConf.load(WORLD_SCORE_ROOT / "config/model_configs/mirage.yaml")
    return OmegaConf.merge(base_config, model_config)


def apply_worldscore_defaults(
    args: argparse.Namespace,
    config: Any,
) -> argparse.Namespace:
    if args.num_frames is None:
        args.num_frames = int(config["frames"])
    if args.fps is None:
        args.fps = int(config["fps"])
    if args.height is None and args.width is None:
        width, height = tuple(int(x) for x in config["resolution"])
        args.height = height
        args.width = width
    return args


def build_intrinsics(
    num_frames: int,
    resolution: tuple[int, int],
    focal_length: float,
) -> np.ndarray:
    width, height = resolution
    intrinsics = np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.repeat(intrinsics[None], num_frames, axis=0)


def convert_worldscore_poses(
    cameras_interp: Any,
    *,
    num_frames: int,
    dry_run: bool,
) -> np.ndarray:
    poses_c2w = cameras_interp
    if isinstance(poses_c2w, torch.Tensor):
        poses_c2w = poses_c2w.detach().cpu().numpy()
    poses_c2w = np.asarray(poses_c2w, dtype=np.float32)
    if poses_c2w.shape[0] < num_frames:
        raise ValueError(
            f"WorldScore provided {poses_c2w.shape[0]} poses, "
            f"expected at least {num_frames}."
        )

    raw = poses_c2w
    # WorldScore stores Blender/OpenGL camera-to-world poses. The latent geometry
    # code expects OpenCV camera axes: x right, y down, z forward.
    blender_to_opencv_camera = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    poses_c2w = poses_c2w @ blender_to_opencv_camera
    if dry_run:
        print("Pose convention conversion (WorldScore Blender c2w -> OpenCV c2w):")
        print(f"  raw R diag: {np.diag(raw[0, :3, :3]).tolist()}")
        print(f"  new R diag: {np.diag(poses_c2w[0, :3, :3]).tolist()}")
        print(f"  raw t: {raw[0, :3, 3].tolist()}")
        print(f"  new t: {poses_c2w[0, :3, 3].tolist()}")
    return poses_c2w[:num_frames]


def load_first_frame(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def build_initial_depth(
    *,
    pipeline: Any,
    frame: np.ndarray,
    intrinsics: np.ndarray,
    poses_c2w: np.ndarray,
) -> np.ndarray:
    geometry = VideoGeometry(
        frames=frame[None],
        depths=np.ones((1, frame.shape[0], frame.shape[1]), dtype=np.float32),
        intrinsics=intrinsics[:1],
        poses_c2w=poses_c2w[:1],
        original_size=frame.shape[:2],
        processed_size=frame.shape[:2],
    )
    predictions = infer_mapanything_depths(
        images=frame[None],
        geometry=geometry,
        pose_indices=[0],
        model_id=pipeline.config.mapanything_model_id,
        device=torch.device(pipeline.pipe.device),
        model_cache=pipeline,
    )
    return resize_depth_to_hw(predictions[0]["depth"], frame.shape[:2])


def resize_depth_to_hw(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if tuple(depth.shape) == target_hw:
        return depth

    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        depth_tensor,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


def write_worldscore_geometry(
    *,
    path: Path,
    frame: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    poses_c2w: np.ndarray,
) -> None:
    num_frames = poses_c2w.shape[0]
    depths = np.repeat(depth[None], num_frames, axis=0).astype(np.float32)
    np.savez_compressed(
        path,
        frames=frame[None],
        depths=depths,
        intrinsics=intrinsics.astype(np.float32),
        poses_c2w=poses_c2w.astype(np.float32),
        frame_indices=np.arange(num_frames, dtype=np.int32),
        original_size=np.array(frame.shape[:2], dtype=np.int32),
        processed_size=np.array(frame.shape[:2], dtype=np.int32),
    )


def frames_to_pil_list(frames: torch.Tensor | np.ndarray) -> list[Image.Image]:
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"Expected video frames [T,H,W,C], got {frames.shape}.")
    if frames.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB frames in the last dimension, got {frames.shape}."
        )
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return [Image.fromarray(frame) for frame in frames]


def generate(
    *,
    pipeline: Any,
    conditioning_image: str,
    inpainting_prompt_list: list[str],
    cameras_interp: Any,
    helper: Any,
    args: argparse.Namespace,
    config: Any,
) -> list[Image.Image]:
    resolution = tuple(int(x) for x in config["resolution"])
    poses_c2w = convert_worldscore_poses(
        cameras_interp,
        num_frames=args.num_frames,
        dry_run=args.dry_run,
    )
    intrinsics = build_intrinsics(
        num_frames=args.num_frames,
        resolution=resolution,
        focal_length=float(config.get("focal_length", 500)),
    )
    frame = load_first_frame(conditioning_image)
    depth = build_initial_depth(
        pipeline=pipeline,
        frame=frame,
        intrinsics=intrinsics,
        poses_c2w=poses_c2w,
    )

    sample_dir = Path(helper.path)
    geometry_path = sample_dir / "worldscore_geometry.npz"
    write_worldscore_geometry(
        path=geometry_path,
        frame=frame,
        depth=depth,
        intrinsics=intrinsics,
        poses_c2w=poses_c2w,
    )

    prompt = inpainting_prompt_list[0] if inpainting_prompt_list else ""
    frames = pipeline.generate(
        geometry_path=geometry_path,
        prompt=prompt,
        output_dir=sample_dir,
        run_metadata={
            "worldscore_json_file": args.worldscore_json_file,
            "worldscore_conditioning_image": conditioning_image,
            "model_configs": args.model_config,
            "tokenizer_path": args.tokenizer_path,
            "vace_checkpoint": None
            if args.vace_checkpoint is None
            else str(args.vace_checkpoint),
            "lora_checkpoint": None
            if args.lora_checkpoint is None
            else str(args.lora_checkpoint),
        },
    )
    return frames_to_pil_list(frames)


def main(argv: list[str] | None = None) -> None:
    print_banner("MIRAGE LATENT-MEM WORLDSCORE GENERATION")
    parser = build_parser()
    args = parser.parse_args(argv)

    if not check_model(MODEL_NAME):
        raise ValueError(f"Model not exists: {MODEL_NAME}")

    config = load_worldscore_config()
    args = apply_worldscore_defaults(args, config)
    validate_args(args)
    if args.start_frame != 0:
        raise ValueError("WorldScore runner only supports --start-frame 0.")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank}/{world_size} using device {device}")

    pipeline = load_pipeline_from_args(args, device=device)
    generated_count = 0

    for visual_movement in ["static", "dynamic"]:
        dataloader, helper = GetHelpers(
            MODEL_NAME,
            visual_movement,
            args.worldscore_json_file,
        )
        for idx, data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            conditioning_image, inpainting_prompt_list, _, cameras_interp = (
                helper.adapt(data)
            )
            frames = generate(
                pipeline=pipeline,
                conditioning_image=conditioning_image,
                inpainting_prompt_list=inpainting_prompt_list,
                cameras_interp=cameras_interp,
                helper=helper,
                args=args,
                config=config,
            )
            helper.save(frames)
            empty_cache()
            generated_count += 1

            if args.dry_run and generated_count >= 1:
                print(f"Rank {rank}: dry run complete after 1 generated sample")
                return


if __name__ == "__main__":
    main()
