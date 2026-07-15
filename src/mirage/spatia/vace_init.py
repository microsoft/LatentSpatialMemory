from __future__ import annotations

import torch
from torch import nn

from .wan_video_vace import VaceWanModel


def infer_vace_layers(num_dit_layers: int) -> tuple[int, ...]:
    if num_dit_layers == 30:
        return tuple(range(0, num_dit_layers, 2))
    if num_dit_layers == 40:
        return tuple(range(0, num_dit_layers, 5))
    raise ValueError(
        "Cannot infer VACE layers for a DiT with "
        f"{num_dit_layers} layers. Provide a compatible VACE model."
    )


def initialize_vace_zero_hint(vace: VaceWanModel) -> None:
    for block in vace.vace_blocks:
        nn.init.zeros_(block.after_proj.weight)
        nn.init.zeros_(block.after_proj.bias)


def build_scratch_vace_from_dit(
    dit: nn.Module,
    *,
    use_reentrant: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> VaceWanModel:
    blocks = getattr(dit, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("Cannot initialize VACE from a DiT without transformer blocks.")

    first_block = blocks[0]
    patch_size = tuple(int(value) for value in getattr(dit, "patch_size"))
    vace = VaceWanModel(
        vace_layers=infer_vace_layers(len(blocks)),
        vace_in_dim=96,
        patch_size=patch_size,
        has_image_input=bool(getattr(dit, "has_image_input", False)),
        dim=int(getattr(dit, "dim")),
        num_heads=int(getattr(first_block, "num_heads")),
        ffn_dim=int(getattr(first_block, "ffn_dim")),
        eps=float(getattr(first_block.norm1, "eps", 1e-6)),
        use_reentrant=use_reentrant,
    )
    initialize_vace_zero_hint(vace)
    return vace.to(device=device, dtype=dtype)
