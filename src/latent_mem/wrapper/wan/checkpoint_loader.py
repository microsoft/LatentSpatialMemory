from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

CHECKPOINT_FORMAT_VERSION = 3
SUPPORTED_STRUCTURED_FORMAT_VERSIONS = {2, CHECKPOINT_FORMAT_VERSION}
VACE_NAMESPACE_PREFIXES = ("vace_blocks", "vace_patch_embedding")
WRAPPER_PREFIXES = ("base_model.model.", "model.")
LORA_BASE_PREFIX = "base_model.model."
LORA_ADAPTER_PATTERN = re.compile(r"(lora_(?:A|B|embedding_A|embedding_B))\.default\.")


@dataclass(frozen=True)
class GeneratorCheckpointLoadResult:
    source_key: str
    format_version: int
    num_lora_tensors: int
    num_vace_tensors: int


@dataclass(frozen=True)
class LegacyLoRACandidate:
    raw_key: str
    canonical_key: str
    value: Any
    quality_rank: tuple[int, int, int]


def serialize_generator_checkpoint(
    generator: torch.nn.Module,
    *,
    full_state_dict: dict[str, Any] | None = None,
    training_stages: list[str] | None = None,
) -> dict[str, Any]:
    has_lora = hasattr(generator, "peft_config")
    vace_state_dict = collect_vace_state_dict(
        generator, full_state_dict=full_state_dict
    )
    lora_state_dict = {}
    if has_lora:
        lora_state_dict = collect_lora_state_dict(
            generator,
            full_state_dict=full_state_dict,
        )
        vace_lora_keys = _find_vace_lora_keys(lora_state_dict)
        if vace_lora_keys:
            preview = ", ".join(vace_lora_keys[:3])
            raise ValueError(
                "V3 checkpoints must contain backbone-only LoRA weights; "
                f"found VACE LoRA keys: {preview}"
            )

    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "generator": {
            "vace": vace_state_dict,
            "lora": lora_state_dict,
        },
        "meta": {
            "has_lora": has_lora,
            "training_stages": list(training_stages or []),
        },
    }


def collect_vace_state_dict(
    generator: torch.nn.Module,
    *,
    full_state_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if full_state_dict is None:
        return dict(_resolve_vace_target(generator).state_dict())

    vace_state_dict: dict[str, Any] = {}
    for raw_key, value in full_state_dict.items():
        if "lora_" in raw_key:
            continue

        canonical_key = _canonicalize_current_vace_key(raw_key)
        if canonical_key is None:
            continue
        if canonical_key in vace_state_dict:
            raise ValueError(
                f"Collected duplicate VACE weights for canonical key '{canonical_key}'."
            )
        vace_state_dict[canonical_key] = value

    return vace_state_dict


def collect_lora_state_dict(
    generator: torch.nn.Module,
    *,
    full_state_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if full_state_dict is None:
        raw_lora_state_dict = get_peft_model_state_dict(generator)
    else:
        raw_lora_state_dict = get_peft_model_state_dict(
            generator,
            state_dict=full_state_dict,
        )

    lora_state_dict: dict[str, Any] = {}
    for raw_key, value in raw_lora_state_dict.items():
        canonical_key = _canonicalize_current_lora_key(raw_key)
        if canonical_key in lora_state_dict:
            raise ValueError(
                f"Collected duplicate LoRA weights for canonical key '{canonical_key}'."
            )
        lora_state_dict[canonical_key] = value

    return lora_state_dict


def load_generator_checkpoint(
    generator: torch.nn.Module,
    ckpt_path: Path,
) -> GeneratorCheckpointLoadResult:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Generator checkpoint does not exist: {ckpt_path}")

    checkpoint = _load_checkpoint_file(ckpt_path)
    structured_checkpoint, source_key, source_format_version = (
        _load_structured_generator_checkpoint(checkpoint)
    )
    generator_sections = _validate_structured_generator_sections(structured_checkpoint)

    vace_state_dict = generator_sections["vace"]
    lora_state_dict = generator_sections["lora"]
    has_lora_adapters = hasattr(generator, "peft_config")
    _validate_lora_checkpoint_namespace(
        lora_state_dict,
        source_key=source_key,
        source_format_version=source_format_version,
    )

    if not vace_state_dict:
        raise ValueError("Checkpoint does not contain VACE weights.")

    if has_lora_adapters and not lora_state_dict:
        raise ValueError(
            "Checkpoint does not contain LoRA weights, but the current model uses "
            "LoRA adapters."
        )

    if not has_lora_adapters and lora_state_dict:
        raise ValueError(
            "Checkpoint contains LoRA weights, but the current model does not have "
            "LoRA adapters."
        )

    if lora_state_dict:
        runtime_lora_state_dict = _materialize_runtime_lora_state_dict(
            generator,
            lora_state_dict,
        )
        incompat = set_peft_model_state_dict(generator, runtime_lora_state_dict)
        unexpected_keys = list(getattr(incompat, "unexpected_keys", []) or [])
        if unexpected_keys:
            preview = ", ".join(unexpected_keys[:3])
            raise ValueError(
                "LoRA weights are not fully compatible with the current model; "
                f"unexpected keys: {preview}"
            )

    vace_target = _resolve_vace_target(generator)
    target_state_dict = vace_target.state_dict()
    vace_state_dict = _expand_vace_patch_embedding_mask_channel(
        vace_state_dict,
        target_state_dict,
    )
    missing_keys, shape_mismatches = _validate_vace_state_dict(
        vace_state_dict,
        target_state_dict,
    )
    if missing_keys or shape_mismatches:
        issue_preview = []
        if missing_keys:
            issue_preview.append(f"missing keys: {', '.join(missing_keys[:3])}")
        if shape_mismatches:
            mismatch_preview = ", ".join(
                f"{key} (checkpoint {checkpoint_shape} vs model {model_shape})"
                for key, checkpoint_shape, model_shape in shape_mismatches[:3]
            )
            issue_preview.append(f"shape mismatches: {mismatch_preview}")
        raise ValueError(
            "VACE weights are not fully compatible with the current model; "
            + "; ".join(issue_preview)
        )

    incompat = vace_target.load_state_dict(vace_state_dict, strict=False)
    unexpected_keys = list(getattr(incompat, "unexpected_keys", []) or [])
    if unexpected_keys:
        preview = ", ".join(unexpected_keys[:3])
        raise ValueError(
            f"VACE weights produced unexpected keys during load: {preview}"
        )

    return GeneratorCheckpointLoadResult(
        source_key=source_key,
        format_version=source_format_version or CHECKPOINT_FORMAT_VERSION,
        num_lora_tensors=len(lora_state_dict),
        num_vace_tensors=len(vace_state_dict),
    )


def load_legacy_generator_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if "generator_ema" in checkpoint and "generator" not in checkpoint:
        raise ValueError("EMA checkpoints are no longer supported.")

    source_key = "checkpoint"
    source_state_dict = checkpoint

    if "generator" in checkpoint:
        source_state_dict = checkpoint["generator"]
        source_key = "generator"
        if not isinstance(source_state_dict, dict):
            raise ValueError(
                "Legacy checkpoint field 'generator' must be a state dict."
            )

    vace_state_dict = _extract_legacy_vace_state_dict(source_state_dict)
    lora_state_dict = _extract_legacy_lora_state_dict(source_state_dict)

    if not vace_state_dict and not lora_state_dict:
        raise ValueError(
            "Legacy checkpoint does not contain any loadable VACE or LoRA weights."
        )

    training_stages = checkpoint.get("training_stages", [])
    if training_stages is None:
        training_stages = []
    if not isinstance(training_stages, list):
        raise ValueError("Legacy checkpoint field 'training_stages' must be a list.")

    structured_checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "generator": {
            "vace": vace_state_dict,
            "lora": lora_state_dict,
        },
        "meta": {
            "has_lora": bool(lora_state_dict),
            "training_stages": training_stages,
        },
    }
    return structured_checkpoint, f"legacy_{source_key}"


def _load_checkpoint_file(ckpt_path: Path) -> dict[str, Any]:
    if str(ckpt_path).endswith(".safetensors"):
        return load_file(str(ckpt_path), device="cpu")
    return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)


def _load_structured_generator_checkpoint(
    checkpoint: Any,
) -> tuple[dict[str, Any], str, int | None]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Generator checkpoint must deserialize to a dict.")

    format_version = checkpoint.get("format_version")
    if format_version in SUPPORTED_STRUCTURED_FORMAT_VERSIONS:
        if "generator_ema" in checkpoint:
            raise ValueError("EMA checkpoints are no longer supported.")
        return checkpoint, f"format_v{format_version}", format_version

    if "format_version" in checkpoint:
        raise ValueError(
            f"Unsupported checkpoint format_version: {checkpoint['format_version']}"
        )

    structured_checkpoint, source_key = load_legacy_generator_checkpoint(checkpoint)
    return structured_checkpoint, source_key, None


def _validate_structured_generator_sections(
    checkpoint: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    generator_payload = checkpoint.get("generator")
    if not isinstance(generator_payload, dict):
        raise ValueError("Structured checkpoint field 'generator' must be a dict.")

    vace_state_dict = generator_payload.get("vace")
    if not isinstance(vace_state_dict, dict):
        raise ValueError("Structured checkpoint field 'generator.vace' must be a dict.")

    lora_state_dict = generator_payload.get("lora")
    if not isinstance(lora_state_dict, dict):
        raise ValueError("Structured checkpoint field 'generator.lora' must be a dict.")

    return {
        "vace": vace_state_dict,
        "lora": lora_state_dict,
    }


def _extract_legacy_vace_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    vace_state_dict: dict[str, Any] = {}
    for raw_key, value in state_dict.items():
        if "lora_" in raw_key:
            continue

        canonical_key = _canonicalize_legacy_vace_key(raw_key)
        if canonical_key is None:
            continue
        if canonical_key in vace_state_dict:
            raise ValueError(
                "Legacy checkpoint contains duplicate VACE weights for canonical key "
                f"'{canonical_key}'."
            )
        vace_state_dict[canonical_key] = value

    return vace_state_dict


def _extract_legacy_lora_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    grouped_candidates: dict[str, list[LegacyLoRACandidate]] = {}
    for raw_key, value in state_dict.items():
        if "lora_" not in raw_key:
            continue

        canonical_key = _canonicalize_legacy_lora_key(raw_key)
        candidate = LegacyLoRACandidate(
            raw_key=raw_key,
            canonical_key=canonical_key,
            value=value,
            quality_rank=_legacy_lora_quality_rank(raw_key, canonical_key),
        )
        grouped_candidates.setdefault(canonical_key, []).append(candidate)

    lora_state_dict: dict[str, Any] = {}
    for canonical_key, candidates in grouped_candidates.items():
        selected_candidate = _select_legacy_lora_candidate(
            canonical_key,
            candidates,
        )
        lora_state_dict[canonical_key] = selected_candidate.value

    return lora_state_dict


def _validate_lora_checkpoint_namespace(
    lora_state_dict: dict[str, Any],
    *,
    source_key: str,
    source_format_version: int | None,
) -> None:
    vace_lora_keys = _find_vace_lora_keys(lora_state_dict)
    if not vace_lora_keys:
        return

    preview = ", ".join(vace_lora_keys[:3])
    if source_format_version == CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "V3 checkpoints must contain backbone-only LoRA weights; "
            f"found VACE LoRA keys: {preview}"
        )

    raise ValueError(
        f"Checkpoint '{source_key}' contains VACE LoRA weights from an older format. "
        "Please convert it with scripts/convert_lora_checkpoint_to_v3.py before "
        f"loading. Found keys: {preview}"
    )


def _find_vace_lora_keys(lora_state_dict: dict[str, Any]) -> list[str]:
    return [key for key in lora_state_dict if _is_vace_lora_key(key)]


def _is_vace_lora_key(key: str) -> bool:
    return (
        key.startswith("vace_blocks.")
        or key.startswith("vace_patch_embedding.")
        or ".vace_blocks." in key
        or ".vace_patch_embedding." in key
    )


def _materialize_runtime_lora_state_dict(
    generator: torch.nn.Module,
    lora_state_dict: dict[str, Any],
) -> dict[str, Any]:
    runtime_keys, canonical_key_map = _collect_runtime_lora_key_metadata(generator)
    runtime_lora_state_dict: dict[str, Any] = {}
    unresolved_keys: list[str] = []

    for raw_key, value in lora_state_dict.items():
        runtime_key = _resolve_runtime_lora_key(
            raw_key,
            runtime_keys=runtime_keys,
            canonical_key_map=canonical_key_map,
        )
        if runtime_key is None:
            unresolved_keys.append(raw_key)
            continue

        if runtime_key in runtime_lora_state_dict:
            raise ValueError(
                "Checkpoint contains duplicate LoRA weights for runtime key "
                f"'{runtime_key}'."
            )
        runtime_lora_state_dict[runtime_key] = value

    if unresolved_keys:
        preview = ", ".join(unresolved_keys[:3])
        raise ValueError(
            "LoRA weights are not fully compatible with the current model; "
            f"unresolved keys: {preview}"
        )

    return runtime_lora_state_dict


def _collect_runtime_lora_key_metadata(
    generator: torch.nn.Module,
) -> tuple[set[str], dict[str, str]]:
    runtime_state_dict = get_peft_model_state_dict(generator)
    runtime_keys = set(runtime_state_dict)
    canonical_key_map: dict[str, str] = {}

    for runtime_key in runtime_state_dict:
        canonical_key = _canonicalize_current_lora_key(runtime_key)
        existing_key = canonical_key_map.get(canonical_key)
        if existing_key is not None and existing_key != runtime_key:
            raise ValueError(
                "Current model exposes multiple LoRA runtime keys for canonical key "
                f"'{canonical_key}': {existing_key}, {runtime_key}"
            )
        canonical_key_map[canonical_key] = runtime_key

    return runtime_keys, canonical_key_map


def _resolve_runtime_lora_key(
    key: str,
    *,
    runtime_keys: set[str],
    canonical_key_map: dict[str, str],
) -> str | None:
    if key in runtime_keys:
        return key

    canonical_key = _canonicalize_current_lora_key(key)
    return canonical_key_map.get(canonical_key)


def _expand_vace_patch_embedding_mask_channel(
    vace_state_dict: dict[str, Any],
    target_state_dict: dict[str, Any],
) -> dict[str, Any]:
    key = "vace_patch_embedding.weight"
    value = vace_state_dict.get(key)
    target_value = target_state_dict.get(key)
    if value is None or target_value is None or value.shape == target_value.shape:
        return vace_state_dict

    can_expand_mask_channel = (
        value.ndim == 5
        and target_value.ndim == 5
        and value.shape[0] == target_value.shape[0]
        and target_value.shape[1] == value.shape[1] + 1
        and value.shape[2:] == target_value.shape[2:]
    )
    if not can_expand_mask_channel:
        return vace_state_dict

    expanded_value = value.new_zeros(target_value.shape)
    expanded_value[:, : value.shape[1]].copy_(value)

    compatible_state_dict = dict(vace_state_dict)
    compatible_state_dict[key] = expanded_value
    return compatible_state_dict


def _canonicalize_current_vace_key(key: str) -> str | None:
    candidate = key
    while True:
        if _has_vace_namespace(candidate):
            return candidate

        stripped = False
        for prefix in WRAPPER_PREFIXES:
            if candidate.startswith(prefix):
                candidate = candidate.removeprefix(prefix)
                stripped = True
                break

        if not stripped:
            return None


def _canonicalize_current_lora_key(key: str) -> str:
    candidate = key
    while candidate.startswith("model."):
        candidate = candidate.removeprefix("model.")

    if candidate.startswith("base_model.model.model."):
        candidate = candidate.replace("base_model.model.model.", LORA_BASE_PREFIX, 1)

    return LORA_ADAPTER_PATTERN.sub(r"\1.", candidate)


def _canonicalize_legacy_vace_key(key: str) -> str | None:
    return _canonicalize_current_vace_key(key)


def _canonicalize_legacy_lora_key(key: str) -> str:
    candidate = key
    while candidate.startswith("model."):
        candidate = candidate.removeprefix("model.")

    if candidate.startswith("base_model.model.model."):
        candidate = candidate.replace(
            "base_model.model.model.",
            LORA_BASE_PREFIX,
            1,
        )

    candidate = LORA_ADAPTER_PATTERN.sub(r"\1.", candidate)
    return candidate


def _legacy_lora_quality_rank(raw_key: str, canonical_key: str) -> tuple[int, int, int]:
    is_canonical = int(raw_key == canonical_key)
    wrapper_penalty = 0
    if raw_key.startswith("model."):
        wrapper_penalty += 1
    if "base_model.model.model." in raw_key:
        wrapper_penalty += 1
    adapter_depth = raw_key.count(".default.")

    return (is_canonical, -wrapper_penalty, -adapter_depth)


def _select_legacy_lora_candidate(
    canonical_key: str,
    candidates: list[LegacyLoRACandidate],
) -> LegacyLoRACandidate:
    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: candidate.quality_rank,
        reverse=True,
    )
    best_rank = ranked_candidates[0].quality_rank
    best_candidates = [
        candidate
        for candidate in ranked_candidates
        if candidate.quality_rank == best_rank
    ]

    reference_candidate = best_candidates[0]
    for candidate in best_candidates[1:]:
        if candidate.value.shape != reference_candidate.value.shape:
            raise ValueError(
                "Legacy checkpoint contains conflicting LoRA shapes for canonical key "
                f"'{canonical_key}'. Raw keys: "
                f"{', '.join(item.raw_key for item in best_candidates[:3])}"
            )
        if not torch.equal(candidate.value, reference_candidate.value):
            raise ValueError(
                "Legacy checkpoint contains conflicting LoRA tensors for canonical key "
                f"'{canonical_key}'. Raw keys: "
                f"{', '.join(item.raw_key for item in best_candidates[:3])}"
            )

    return reference_candidate


def _has_vace_namespace(key: str) -> bool:
    return any(
        key == prefix or key.startswith(f"{prefix}.")
        for prefix in VACE_NAMESPACE_PREFIXES
    )


def _resolve_vace_target(generator: torch.nn.Module) -> torch.nn.Module:
    seen: set[int] = set()
    modules_to_visit = [generator]

    while modules_to_visit:
        module = modules_to_visit.pop()
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)

        if hasattr(module, "vace_blocks") or hasattr(module, "vace_patch_embedding"):
            return module

        for attr_name in ("model", "base_model"):
            child = getattr(module, attr_name, None)
            if isinstance(child, torch.nn.Module):
                modules_to_visit.append(child)

    raise ValueError("Current generator does not contain a VACE target module.")


def _validate_vace_state_dict(
    vace_state_dict: dict[str, Any],
    target_state_dict: dict[str, Any],
) -> tuple[list[str], list[tuple[str, list[int], list[int]]]]:
    missing_keys: list[str] = []
    shape_mismatches: list[tuple[str, list[int], list[int]]] = []

    for key, value in vace_state_dict.items():
        if key not in target_state_dict:
            missing_keys.append(key)
            continue
        if target_state_dict[key].shape != value.shape:
            shape_mismatches.append(
                (
                    key,
                    list(value.shape),
                    list(target_state_dict[key].shape),
                )
            )

    return missing_keys, shape_mismatches
