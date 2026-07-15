#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from diffsynth.core import ModelConfig

from mirage.inference.mirage_pipeline import (
    MirageConfig,
    MiragePipeline,
    load_lora_checkpoint,
    load_vace_checkpoint,
    validate_pipe,
)
from mirage.spatia.vace_init import build_scratch_vace_from_dit
from mirage.spatia.wan_video_new import WanVideoPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        action="append",
        required=True,
        help=(
            "Repeatable DiffSynth ModelConfig. Accepts a local path, "
        ),
    )
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--vace-checkpoint", type=Path, default=None)
    parser.add_argument("--lora-checkpoint", type=Path, default=None)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-path", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--infer-steps", type=int, default=20)
    parser.add_argument("--torch-dtype", type=str, default="bf16")
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--timestep-shift", type=float, default=5.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--no-cfg", dest="no_cfg", action="store_true", default=True)
    parser.add_argument("--use-cfg", dest="no_cfg", action="store_false")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--max-reference-frames", type=int, default=4)
    parser.add_argument("--preceding-pixel-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--mapanything-model-id", type=str, default="facebook/map-anything"
    )
    parser.add_argument("--ref-iou-threshold", type=float, default=0.04)
    parser.add_argument("--ref-iou-voxel-size", type=float, default=0.1)
    parser.add_argument("--no-tiled", action="store_true")
    parser.add_argument("--tile-size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile-stride", type=int, nargs=2, default=(15, 26))
    return parser.parse_args()


def parse_torch_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def parse_model_config(spec: str) -> ModelConfig | tuple[dict[str, Any], ModelConfig]:
    raw = parse_model_config_spec(spec)
    extra_kwargs = raw.pop("extra_kwargs", None) or raw.pop(
        "class_overwrite_kwargs",
        None,
    )
    for key, value in list(raw.items()):
        if key.endswith("_dtype") and isinstance(value, str):
            raw[key] = parse_torch_dtype(value)
    model_config = ModelConfig(**raw)
    if extra_kwargs is not None:
        return extra_kwargs, model_config
    return model_config


def parse_model_config_spec(spec: str) -> dict[str, Any]:
    spec = spec.strip()
    if not spec:
        raise ValueError("--model-config cannot be empty")

    if spec.startswith("{"):
        parsed = json.loads(spec)
        if not isinstance(parsed, dict):
            raise ValueError(f"JSON model config must be an object: {spec}")
        return parsed

    if "=" in spec:
        config = {}
        for item in spec.split(","):
            key, value = item.split("=", 1)
            config[key.strip()] = parse_scalar_value(value.strip())
        return config

    path = Path(spec)
    if path.exists() or spec.endswith((".pt", ".pth", ".bin", ".ckpt", ".safetensors")):
        return {"path": spec}

    if ":" in spec:
        model_id, origin_file_pattern = spec.split(":", 1)
        return {
            "model_id": model_id,
            "origin_file_pattern": origin_file_pattern,
        }

    return {"path": spec}


def parse_scalar_value(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"none", "None", "null"}:
        return None
    return value


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None and args.prompt_path is not None:
        raise ValueError("Use either --prompt or --prompt-path, not both.")
    if args.prompt_path is not None:
        return args.prompt_path.read_text(encoding="utf-8").strip()
    if args.prompt is None:
        raise ValueError("Either --prompt or --prompt-path is required.")
    return args.prompt


def validate_args(args: argparse.Namespace) -> None:
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive.")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative.")
    if args.infer_steps <= 0:
        raise ValueError("--infer-steps must be positive.")
    if args.num_train_timesteps <= 0:
        raise ValueError("--num-train-timesteps must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.max_reference_frames < 0:
        raise ValueError("--max-reference-frames must be non-negative.")
    if args.preceding_pixel_frames < 0:
        raise ValueError("--preceding-pixel-frames must be non-negative.")


def build_inference_config(args: argparse.Namespace) -> MirageConfig:
    return MirageConfig(
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        infer_steps=args.infer_steps,
        num_train_timesteps=args.num_train_timesteps,
        timestep_shift=args.timestep_shift,
        guidance_scale=args.guidance_scale,
        no_cfg=args.no_cfg,
        fps=args.fps,
        max_reference_frames=args.max_reference_frames,
        preceding_pixel_frames=args.preceding_pixel_frames,
        seed=args.seed,
        height=args.height,
        width=args.width,
        tiled=not args.no_tiled,
        tile_size=tuple(args.tile_size),
        tile_stride=tuple(args.tile_stride),
        mapanything_model_id=args.mapanything_model_id,
        ref_iou_threshold=args.ref_iou_threshold,
        ref_iou_voxel_size=args.ref_iou_voxel_size,
    )


def load_pipeline_from_args(
    args: argparse.Namespace,
    *,
    device: torch.device | None = None,
) -> MiragePipeline:
    runtime_dtype = parse_torch_dtype(args.torch_dtype)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_configs = [parse_model_config(spec) for spec in args.model_config]
    tokenizer_config = ModelConfig(path=args.tokenizer_path)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=runtime_dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    if pipe.vace is None and args.vace_checkpoint is not None:
        if pipe.dit is None:
            validate_pipe(pipe)
        pipe.vace = build_scratch_vace_from_dit(
            pipe.dit,
            use_reentrant=False,
            device=device,
            dtype=runtime_dtype,
        )
        print(
            "Initialized scratch VACE: "
            f"layers={pipe.vace.vace_layers} "
            f"dim={pipe.vace.vace_patch_embedding.out_channels} "
            f"channels={pipe.vace.vace_patch_embedding.in_channels}"
        )

    if args.vace_checkpoint is not None:
        load_vace_checkpoint(pipe, args.vace_checkpoint)
    validate_pipe(pipe)

    if args.lora_checkpoint is not None:
        load_lora_checkpoint(pipe, args.lora_checkpoint, alpha=args.lora_alpha)

    return MiragePipeline(pipe, build_inference_config(args))


def main() -> None:
    args = parse_args()
    validate_args(args)
    prompt = read_prompt(args)
    pipeline = load_pipeline_from_args(args)
    pipeline.generate(
        geometry_path=args.geometry_path,
        prompt=prompt,
        output_dir=args.output_dir,
        run_metadata={
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


if __name__ == "__main__":
    main()
