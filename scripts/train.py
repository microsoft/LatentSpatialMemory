#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Avoid allocator fragmentation in long video training unless the launcher
# already provides a CUDA allocator policy.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffsynth.core import ModelConfig
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file as load_safetensors
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mirage.dataset import DataConfig, MirageDataset, mirage_collate_fn  # noqa: E402
from mirage.spatia.wan_video_new import (  # noqa: E402
    WanVideoPipeline,
    model_fn_wan_video,
)
from mirage.spatia.vace_init import build_scratch_vace_from_dit  # noqa: E402

logger = get_logger(__name__)

LOSS_HISTORY_NAME = "loss_history.json"


def validate_diffsynth_2x_api() -> None:
    try:
        from diffsynth.core import ModelConfig as _ModelConfig
        from diffsynth.diffusion import FlowMatchScheduler
        from diffsynth.diffusion.base_pipeline import (
            BasePipeline,
            PipelineUnit,
            PipelineUnitRunner,
        )
    except ImportError as exc:
        raise ImportError(
            "This trainer requires diffsynth 2.x APIs. Install/use "
            "`diffsynth==2.0.12` or another compatible 2.x release."
        ) from exc

    required_symbols = {
        "diffsynth.core.ModelConfig": _ModelConfig,
        "diffsynth.diffusion.FlowMatchScheduler": FlowMatchScheduler,
        "diffsynth.diffusion.base_pipeline.BasePipeline": BasePipeline,
        "diffsynth.diffusion.base_pipeline.PipelineUnit": PipelineUnit,
        "diffsynth.diffusion.base_pipeline.PipelineUnitRunner": PipelineUnitRunner,
    }
    missing = [name for name, symbol in required_symbols.items() if symbol is None]
    if missing:
        raise ImportError(
            "The installed diffsynth package is missing required 2.x APIs: "
            + ", ".join(missing)
        )


class MirageTrainingModel(nn.Module):
    def __init__(
        self,
        dit: nn.Module,
        vace: nn.Module,
        *,
        gradient_checkpointing: bool,
        use_reentrant: bool,
    ):
        super().__init__()
        self.dit = dit
        self.vace = vace
        self.gradient_checkpointing = gradient_checkpointing
        self.use_reentrant = use_reentrant

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        vace_context: torch.Tensor,
        num_ref_frames: int,
    ) -> torch.Tensor:
        return model_fn_wan_video(
            dit=self.dit,
            vace=self.vace,
            latents=latents,
            timestep=timestep,
            context=context,
            vace_context=vace_context,
            vace_scale=1.0,
            num_ref_frames=num_ref_frames,
            use_gradient_checkpointing=self.gradient_checkpointing,
            use_reentrant=self.use_reentrant,
            fuse_vae_embedding_in_latents=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the DiffSynth inference model on latent projection LMDB data."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        action="append",
        required=True,
        help=(
            "Repeatable DiffSynth ModelConfig. Accepts a local path, "
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Optional local tokenizer path. If omitted, WanVideoPipeline default is used.",
    )
    parser.add_argument(
        "--init-vace-checkpoint",
        type=Path,
        default=None,
        help="Optional VACE-only initialization checkpoint",
    )
    parser.add_argument(
        "--init-lora-checkpoint",
        type=Path,
        default=None,
        help="Optional LoRA-only initialization checkpoint",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="vace,lora",
        help="Comma-separated stages. Supported values: vace,lora.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
        help="Optimizer steps per stage.",
    )
    parser.add_argument("--lr-vace", type=float, default=1e-5)
    parser.add_argument("--lr-lora", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bf16",
        help="Model load dtype: bf16, fp16, or fp32.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation checkpointing to reduce training memory.",
    )
    parser.add_argument("--use-reentrant", action="store_true")
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--timestep-shift", type=float, default=5.0)
    parser.add_argument("--drop-text-prompt", type=float, default=0.1)
    parser.add_argument("--random-sample-ref", action="store_true")
    parser.add_argument("--random-sample-preceding", action="store_true")
    parser.add_argument("--max-reference-frames", type=int, default=0)
    parser.add_argument("--max-preceding-frames", type=int, default=0)
    parser.add_argument(
        "--model-version",
        choices=("5b", "14b"),
        default="5b",
        help="LMDB reference latent layout.",
    )
    parser.add_argument(
        "--keep-first-frame-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep target latent frame 0 clean and exclude it from loss.",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Comma-separated PEFT target module names.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
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


def validate_numeric_args(args: argparse.Namespace) -> None:
    positive_int_fields = (
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "log_steps",
        "save_steps",
        "num_train_timesteps",
    )
    for field in positive_int_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.max_reference_frames < 0:
        raise ValueError("--max-reference-frames must be non-negative.")
    if args.max_preceding_frames < 0:
        raise ValueError("--max-preceding-frames must be non-negative.")
    if not 0 <= args.drop_text_prompt <= 1:
        raise ValueError("--drop-text-prompt must be in [0, 1].")


def move_pipe_conditioning_models(
    pipe: WanVideoPipeline,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device=device, dtype=dtype)
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder.eval()
    if pipe.image_encoder is not None:
        pipe.image_encoder.to(device=device, dtype=dtype)
        pipe.image_encoder.requires_grad_(False)
        pipe.image_encoder.eval()
    if pipe.vae is not None:
        pipe.vae.requires_grad_(False)
        pipe.vae.eval()


def validate_fused_latent_dit(dit: nn.Module) -> None:
    if not getattr(dit, "fuse_vae_embedding_in_latents", False):
        raise ValueError(
            "This trainer expects a fused-latent DiT; "
            "loaded DiT has fuse_vae_embedding_in_latents=False."
        )
    if not getattr(dit, "seperated_timestep", False):
        raise ValueError(
            "This trainer expects per-frame separated timesteps; "
            "loaded DiT has seperated_timestep=False."
        )


def parse_model_config(spec: str) -> ModelConfig | tuple[dict[str, Any], ModelConfig]:
    raw = parse_model_config_spec(spec)
    extra_kwargs = raw.pop("extra_kwargs", None) or raw.pop(
        "class_overwrite_kwargs", None
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
        config: dict[str, Any] = {}
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


def build_tokenizer_config(path: str | None) -> ModelConfig | None:
    if path is None:
        return None
    return ModelConfig(path=path)


def apply_lora_to_dit(dit: nn.Module, args: argparse.Namespace) -> nn.Module:
    target_modules = [
        item.strip() for item in args.lora_target_modules.split(",") if item.strip()
    ]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    peft_dit = get_peft_model(dit, lora_config, autocast_adapter_dtype=False)
    for _, config in peft_dit.peft_config.items():
        config.inference_mode = False
    return get_peft_forward_model(peft_dit)


def get_peft_forward_model(model: nn.Module) -> nn.Module:
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        return base_model.model
    return model


def mark_potential_trainable_parameters(model: nn.Module, stages: list[str]) -> None:
    model.requires_grad_(False)
    if "vace" in stages:
        model.vace.requires_grad_(True)
    if "lora" in stages:
        for name, param in model.dit.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)


def normalize_floating_dtype(model: nn.Module, dtype: torch.dtype) -> None:
    for param in model.parameters():
        if param.is_floating_point() and param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype=dtype)
    for buffer in model.buffers():
        if buffer.is_floating_point() and buffer.dtype != dtype:
            buffer.data = buffer.data.to(dtype=dtype)


def set_stage_trainable(model: nn.Module, stage: str) -> list[nn.Parameter]:
    model.requires_grad_(False)
    if stage == "vace":
        model.vace.train()
        model.dit.eval()
        for param in model.vace.parameters():
            param.requires_grad_(True)
    elif stage == "lora":
        model.vace.eval()
        model.dit.train()
        for name, param in model.dit.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
    else:
        raise ValueError(f"Unsupported training stage: {stage}")

    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError(f"Stage '{stage}' did not select any trainable parameters.")
    return params


def load_safetensors_checkpoint(path: Path) -> dict[str, Any]:
    if path.suffix != ".safetensors":
        raise ValueError(f"Only .safetensors checkpoints are supported: {path}")
    return dict(load_safetensors(str(path), device="cpu"))


def load_vace_weights(model: nn.Module, checkpoint: dict[str, Any], path: Path) -> None:
    vace_state = extract_vace_state_dict(checkpoint)
    if not vace_state:
        logger.info("No VACE tensors found in %s", path)
        return
    vace_state, skipped = filter_compatible_state_dict(
        vace_state,
        model.vace.state_dict(),
    )
    if skipped:
        logger.info("Skipped %d VACE tensors with incompatible shapes.", skipped)
    missing, unexpected = model.vace.load_state_dict(vace_state, strict=False)
    logger.info(
        "Loaded VACE weights from %s: %d tensors, %d missing, %d unexpected",
        path,
        len(vace_state),
        len(missing),
        len(unexpected),
    )


def load_lora_weights(model: nn.Module, checkpoint: dict[str, Any], path: Path) -> None:
    lora_state = extract_lora_state_dict(checkpoint, model.dit.state_dict())
    if not lora_state:
        logger.info("No LoRA tensors found in %s", path)
        return
    lora_state, skipped = filter_compatible_state_dict(
        lora_state,
        model.dit.state_dict(),
    )
    if skipped:
        logger.info("Skipped %d LoRA tensors with incompatible shapes.", skipped)
    missing, unexpected = model.dit.load_state_dict(lora_state, strict=False)
    logger.info(
        "Loaded LoRA weights from %s: %d tensors, %d missing, %d unexpected",
        path,
        len(lora_state),
        len(missing),
        len(unexpected),
    )


def filter_compatible_state_dict(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    compatible = {}
    skipped = 0
    for key, value in source.items():
        if key not in target or value.shape != target[key].shape:
            skipped += 1
            continue
        target_tensor = target[key]
        if value.is_floating_point() and value.dtype != target_tensor.dtype:
            value = value.to(dtype=target_tensor.dtype)
        compatible[key] = value
    return compatible, skipped


def extract_vace_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    candidate = None
    if isinstance(checkpoint.get("vace"), dict):
        candidate = checkpoint["vace"]
    elif isinstance(checkpoint.get("generator"), dict) and isinstance(
        checkpoint["generator"].get("vace"), dict
    ):
        candidate = checkpoint["generator"]["vace"]
    elif isinstance(checkpoint.get("state_dict"), dict):
        candidate = checkpoint["state_dict"]
    else:
        candidate = checkpoint

    state: dict[str, torch.Tensor] = {}
    for key, value in candidate.items():
        if not torch.is_tensor(value):
            continue
        normalized = strip_prefixes(
            key,
            ("module.", "model.", "pipe.vace.", "vace."),
        )
        if normalized.startswith(("vace_blocks.", "vace_patch_embedding.")):
            state[normalized] = value
    return state


def extract_lora_state_dict(
    checkpoint: dict[str, Any],
    target_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    candidate = None
    if isinstance(checkpoint.get("lora"), dict):
        candidate = checkpoint["lora"]
    elif isinstance(checkpoint.get("generator"), dict) and isinstance(
        checkpoint["generator"].get("lora"), dict
    ):
        candidate = checkpoint["generator"]["lora"]
    elif isinstance(checkpoint.get("state_dict"), dict):
        candidate = checkpoint["state_dict"]
    else:
        candidate = checkpoint

    state: dict[str, torch.Tensor] = {}
    for key, value in candidate.items():
        if not torch.is_tensor(value) or "lora_" not in key:
            continue
        normalized = normalize_lora_key(key)
        mapped_key = map_lora_key_to_target(normalized, target_state)
        if mapped_key is not None:
            state[mapped_key] = value
    return state


def strip_prefixes(key: str, prefixes: tuple[str, ...]) -> str:
    normalized = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def normalize_lora_key(key: str) -> str:
    normalized = strip_prefixes(
        key,
        (
            "module.",
            "model.",
            "pipe.dit.",
            "dit.",
            "base_model.model.",
            "base_model.",
        ),
    )
    normalized = normalized.replace(".lora_A.weight", ".lora_A.default.weight")
    normalized = normalized.replace(".lora_B.weight", ".lora_B.default.weight")
    return normalized


def map_lora_key_to_target(
    key: str,
    target_state: dict[str, torch.Tensor],
) -> str | None:
    if key in target_state:
        return key
    for target_key in target_state:
        if "lora_" in target_key and target_key.endswith(key):
            return target_key
    suffix = key.replace(".default.weight", ".weight")
    for target_key in target_state:
        if "lora_" in target_key and target_key.endswith(suffix):
            return target_key
    return None


def build_dataloader(args: argparse.Namespace) -> DataLoader:
    data_config = DataConfig(
        data_path=args.data_path,
        random_sample_ref=args.random_sample_ref,
        random_sample_preceding=args.random_sample_preceding,
        max_reference_frames=args.max_reference_frames,
        max_preceding_frames=args.max_preceding_frames,
    )
    dataset = MirageDataset(data_config, model_version=args.model_version)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=mirage_collate_fn,
    )


def maybe_drop_prompts(prompts: list[str], drop_probability: float) -> list[str]:
    if drop_probability <= 0:
        return prompts
    return ["" if random.random() < drop_probability else prompt for prompt in prompts]


def trim_variable_frames(
    tensor: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return tensor
    count = int(mask.sum(dim=1).min().item())
    return tensor[:, :count]


def prepare_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    expected_vace_channels: int,
    vae: nn.Module,
) -> dict[str, Any]:
    target = batch["target_latent"].to(device=device, dtype=dtype)
    target_scene = batch["target_scene_proj"].to(device=device, dtype=dtype)

    preceding = trim_variable_frames(
        batch["preceding_latent"].to(device=device, dtype=dtype),
        batch.get("preceding_mask"),
    )
    preceding_scene = trim_variable_frames(
        batch["preceding_scene_proj"].to(device=device, dtype=dtype),
        batch.get("preceding_mask"),
    )
    reference = trim_variable_frames(
        batch["reference_latent"].to(device=device, dtype=dtype),
        batch.get("reference_mask"),
    )

    vace_context = build_vace_context(
        target_scene=target_scene,
        preceding_scene=preceding_scene,
        expected_channels=expected_vace_channels,
        vae=vae,
        device=device,
        dtype=dtype,
    )

    return {
        "target": target.permute(0, 2, 1, 3, 4).contiguous(),
        "preceding": preceding.permute(0, 2, 1, 3, 4).contiguous(),
        "reference": reference.permute(0, 2, 1, 3, 4).contiguous(),
        "vace_context": vace_context,
    }


def build_vace_context(
    *,
    target_scene: torch.Tensor,
    preceding_scene: torch.Tensor,
    expected_channels: int,
    vae: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    scene = torch.cat([target_scene, preceding_scene], dim=1)
    current_channels = scene.shape[2]
    if current_channels == 48 and expected_channels == 96:
        score_latent = build_control_score_latent(
            scene=scene,
            vae=vae,
            device=device,
            dtype=dtype,
        )
        scene = torch.cat([scene, score_latent], dim=2)
    elif current_channels + 1 == expected_channels:
        hole_mask = (~scene.ne(0).any(dim=2, keepdim=True)).to(dtype=scene.dtype)
        scene = torch.cat([scene, hole_mask], dim=2)
    elif current_channels != expected_channels:
        raise ValueError(
            "VACE context channel mismatch: "
            f"model expects {expected_channels}, data has {current_channels}."
        )
    return scene.permute(0, 2, 1, 3, 4).contiguous()


def build_control_score_latent(
    *,
    scene: torch.Tensor,
    vae: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, latent_frames, _, latent_height, latent_width = scene.shape
    pixel_frames = (latent_frames - 1) * 4 + 1
    upsampling_factor = int(vae.upsampling_factor)
    pixel_height = latent_height * upsampling_factor
    pixel_width = latent_width * upsampling_factor
    score_video = torch.ones(
        batch_size,
        3,
        pixel_frames,
        pixel_height,
        pixel_width,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        score_latent = vae.encode(
            score_video,
            device=device,
            tiled=False,
        ).to(device=device, dtype=dtype)
    expected_shape = (
        batch_size,
        48,
        latent_frames,
        latent_height,
        latent_width,
    )
    if tuple(score_latent.shape) != expected_shape:
        raise ValueError(
            "Dummy control score latent shape mismatch: "
            f"expected {expected_shape}, got {tuple(score_latent.shape)}."
        )
    return score_latent.permute(0, 2, 1, 3, 4).contiguous()


def get_expected_vace_channels(model: nn.Module) -> int:
    patch_embedding = getattr(model.vace, "vace_patch_embedding", None)
    if patch_embedding is None:
        raise ValueError("VACE model is missing vace_patch_embedding.")
    return int(patch_embedding.in_channels)


def sample_flow_timesteps(
    *,
    target_frames: int,
    num_train_timesteps: int,
    shift: float,
    device: torch.device,
    keep_first_frame_clean: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = torch.rand(target_frames, device=device, dtype=torch.float32)
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    timestep = sigma * num_train_timesteps
    if keep_first_frame_clean:
        sigma[0] = 0
        timestep[0] = 0
    return sigma, timestep


def compute_flow_matching_loss(
    model: nn.Module,
    batch: dict[str, Any],
    context: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    target = batch["target"]
    reference = batch["reference"]
    preceding = batch["preceding"]

    batch_size, channels, target_frames, height, width = target.shape
    if args.keep_first_frame_clean and target_frames <= 1:
        raise ValueError(
            "Cannot exclude first frame from loss when target has <= 1 frame."
        )

    sigma, target_timestep = sample_flow_timesteps(
        target_frames=target_frames,
        num_train_timesteps=args.num_train_timesteps,
        shift=args.timestep_shift,
        device=target.device,
        keep_first_frame_clean=args.keep_first_frame_clean,
    )
    noise = torch.randn(
        (batch_size, channels, target_frames, height, width),
        device=target.device,
        dtype=target.dtype,
    )
    sigma_view = sigma.to(dtype=target.dtype).view(1, 1, target_frames, 1, 1)
    noisy_target = (1 - sigma_view) * target + sigma_view * noise
    if args.keep_first_frame_clean:
        noisy_target[:, :, 0].copy_(target[:, :, 0])

    latents = torch.cat([reference, noisy_target, preceding], dim=2)
    timestep = build_full_timestep(
        num_ref=reference.shape[2],
        target_timestep=target_timestep,
        num_pre=preceding.shape[2],
    )

    flow_pred = model(
        latents=latents,
        timestep=timestep,
        context=context,
        vace_context=batch["vace_context"],
        num_ref_frames=reference.shape[2],
    )
    start = reference.shape[2]
    stop = start + target_frames
    flow_pred_target = flow_pred[:, :, start:stop]
    flow_target = noise - target
    if args.keep_first_frame_clean:
        flow_pred_target = flow_pred_target[:, :, 1:]
        flow_target = flow_target[:, :, 1:]
    return F.mse_loss(flow_pred_target.float(), flow_target.float())


def build_full_timestep(
    *,
    num_ref: int,
    target_timestep: torch.Tensor,
    num_pre: int,
) -> torch.Tensor:
    pieces = []
    if num_ref > 0:
        pieces.append(target_timestep.new_zeros(num_ref))
    pieces.append(target_timestep)
    if num_pre > 0:
        pieces.append(target_timestep.new_zeros(num_pre))
    return torch.cat(pieces, dim=0)


def to_pipeline_lora_key(key: str) -> str:
    normalized = strip_prefixes(
        key,
        (
            "module.",
            "model.",
            "_fsdp_wrapped_module.",
            "base_model.model.",
            "base_model.",
            "pipe.dit.",
            "dit.",
        ),
    )
    normalized = normalized.replace(".lora_A.default.weight", ".lora_A.weight")
    normalized = normalized.replace(".lora_B.default.weight", ".lora_B.weight")
    return f"pipe.dit.{normalized}"


def export_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous()


def build_export_state(
    state_dict: dict[str, torch.Tensor],
    stage: str,
) -> dict[str, torch.Tensor]:
    if stage == "vace":
        export_state = {}
        for key, value in state_dict.items():
            normalized = strip_prefixes(
                key,
                ("module.", "model.", "_fsdp_wrapped_module.", "vace."),
            )
            if normalized.startswith(("vace_blocks.", "vace_patch_embedding.")):
                export_state[normalized] = export_tensor(value)
        return export_state
    if stage == "lora":
        return {
            to_pipeline_lora_key(key): export_tensor(value)
            for key, value in state_dict.items()
            if "lora_" in key
        }
    raise ValueError(f"Unsupported training stage: {stage}")


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    args: argparse.Namespace,
    stage: str,
    stage_step: int,
) -> None:
    state_dict = accelerator.get_state_dict(model)
    export_state = build_export_state(state_dict, stage)

    if accelerator.is_main_process:
        logger.info(
            "Saving pipeline %s weights: step=%d tensors=%d",
            stage,
            stage_step,
            len(export_state),
        )
        stage_dir = args.output_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = stage_dir / f"step_{stage_step:07d}_{stage}.safetensors"
        accelerator.save(export_state, checkpoint_path, safe_serialization=True)
        logger.info("Saved pipeline %s weights: %s", stage, checkpoint_path)

    accelerator.wait_for_everyone()


def get_loss_history_path(args: argparse.Namespace) -> Path:
    return args.output_dir / LOSS_HISTORY_NAME


def save_loss_history(path: Path, history: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def init_loss_history(
    *,
    accelerator: Accelerator,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not accelerator.is_main_process:
        return []

    path = get_loss_history_path(args)
    save_loss_history(path, [])
    return []


def should_save_final_checkpoint(stage_step: int, save_steps: int) -> bool:
    return stage_step % save_steps != 0


def train_stage(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    dataloader: DataLoader,
    pipe: WanVideoPipeline,
    args: argparse.Namespace,
    stage: str,
    global_step: int,
    runtime_dtype: torch.dtype,
    loss_history: list[dict[str, Any]],
) -> int:
    unwrapped_model = accelerator.unwrap_model(model)
    trainable_params = set_stage_trainable(unwrapped_model, stage)
    lr = args.lr_vace if stage == "vace" else args.lr_lora
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    expected_vace_channels = get_expected_vace_channels(unwrapped_model)
    progress = tqdm(
        total=args.max_steps,
        initial=0,
        disable=not accelerator.is_main_process,
        desc=f"stage={stage}",
        dynamic_ncols=True,
    )
    data_iter = iter(dataloader)
    stage_step = 0
    running_loss = 0.0

    while stage_step < args.max_steps:
        try:
            raw_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            raw_batch = next(data_iter)

        with accelerator.accumulate(model):
            prompts = maybe_drop_prompts(raw_batch["prompts"], args.drop_text_prompt)
            with torch.no_grad():
                context = pipe.prompter.encode_prompt(
                    prompts,
                    positive=True,
                    device=accelerator.device,
                )
                context = context.to(
                    device=accelerator.device,
                    dtype=runtime_dtype,
                )

            batch = prepare_batch(
                raw_batch,
                device=accelerator.device,
                dtype=runtime_dtype,
                expected_vace_channels=expected_vace_channels,
                vae=pipe.vae,
            )
            loss = compute_flow_matching_loss(model, batch, context, args)
            accelerator.backward(loss)
            if accelerator.sync_gradients and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if not accelerator.sync_gradients:
            continue

        loss_value = accelerator.gather(loss.detach()).mean().item()
        running_loss += loss_value
        stage_step += 1
        global_step += 1
        progress.update(1)
        if stage_step % args.log_steps == 0:
            avg_loss = running_loss / args.log_steps
            running_loss = 0.0
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "stage=%s step=%d/%d global_step=%d loss=%.6f lr=%.3e",
                stage,
                stage_step,
                args.max_steps,
                global_step,
                avg_loss,
                lr,
            )
            if accelerator.is_main_process:
                loss_history.append(
                    {
                        "stage": stage,
                        "stage_step": stage_step,
                        "global_step": global_step,
                        "loss": avg_loss,
                        "lr": lr,
                        "log_steps": args.log_steps,
                    }
                )
                save_loss_history(get_loss_history_path(args), loss_history)
        if stage_step % args.save_steps == 0:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                args=args,
                stage=stage,
                stage_step=stage_step,
            )

    progress.close()
    if should_save_final_checkpoint(stage_step, args.save_steps):
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            args=args,
            stage=stage,
            stage_step=stage_step,
        )
    return global_step


def validate_stages(stages: list[str]) -> None:
    supported = {"vace", "lora"}
    unknown = [stage for stage in stages if stage not in supported]
    if unknown:
        raise ValueError(
            f"Unsupported stages: {unknown}. Supported: {sorted(supported)}"
        )


def main() -> None:
    validate_diffsynth_2x_api()
    args = parse_args()
    stages = [stage.strip() for stage in args.stage.split(",") if stage.strip()]
    validate_stages(stages)
    validate_numeric_args(args)
    runtime_dtype = parse_torch_dtype(args.torch_dtype)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
        ],
    )
    set_seed(args.seed + accelerator.process_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_configs = [parse_model_config(spec) for spec in args.model_config]
    tokenizer_config = build_tokenizer_config(args.tokenizer_path)
    pipeline_kwargs = {}
    if tokenizer_config is not None:
        pipeline_kwargs["tokenizer_config"] = tokenizer_config

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=runtime_dtype,
        device=accelerator.device,
        model_configs=model_configs,
        use_reentrant=args.use_reentrant,
        **pipeline_kwargs,
    )
    if pipe.dit is None:
        raise ValueError("WanVideoPipeline did not load a DiT model.")
    validate_fused_latent_dit(pipe.dit)
    if pipe.vace is None:
        pipe.vace = build_scratch_vace_from_dit(
            pipe.dit,
            use_reentrant=args.use_reentrant,
            device=accelerator.device,
            dtype=runtime_dtype,
        )
        logger.info(
            "Initialized scratch VACE: layers=%s dim=%d channels=%d",
            pipe.vace.vace_layers,
            pipe.vace.vace_patch_embedding.out_channels,
            pipe.vace.vace_patch_embedding.in_channels,
        )

    pipe.dit.requires_grad_(False)
    pipe.vace.requires_grad_(False)
    move_pipe_conditioning_models(
        pipe,
        device=accelerator.device,
        dtype=runtime_dtype,
    )

    dit = pipe.dit
    if "lora" in stages:
        dit = apply_lora_to_dit(dit, args)

    model = MirageTrainingModel(
        dit=dit,
        vace=pipe.vace,
        gradient_checkpointing=args.gradient_checkpointing,
        use_reentrant=args.use_reentrant,
    )
    mark_potential_trainable_parameters(model, stages)

    if args.init_vace_checkpoint is not None:
        load_vace_weights(
            model,
            load_safetensors_checkpoint(args.init_vace_checkpoint),
            args.init_vace_checkpoint,
        )
    if args.init_lora_checkpoint is not None:
        if "lora" not in stages:
            raise ValueError("--init-lora-checkpoint requires the lora stage.")
        load_lora_weights(
            model,
            load_safetensors_checkpoint(args.init_lora_checkpoint),
            args.init_lora_checkpoint,
        )

    normalize_floating_dtype(model, runtime_dtype)

    dataloader = build_dataloader(args)
    model, dataloader = accelerator.prepare(model, dataloader)

    global_step = 0
    loss_history = init_loss_history(
        accelerator=accelerator,
        args=args,
    )
    accelerator.wait_for_everyone()

    for stage in stages:
        accelerator.print(f"Starting stage '{stage}' for {args.max_steps} steps")
        global_step = train_stage(
            accelerator=accelerator,
            model=model,
            dataloader=dataloader,
            pipe=pipe,
            args=args,
            stage=stage,
            global_step=global_step,
            runtime_dtype=runtime_dtype,
            loss_history=loss_history,
        )
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
