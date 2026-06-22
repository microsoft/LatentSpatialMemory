from __future__ import annotations

import os
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def get_rank_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def shard_items(
    items: Sequence[T], rank: int, world_size: int, mode: str = "interleave"
) -> list[T]:
    """
    Distribute items across ranks.

    Args:
        items: Sequence of items to distribute
        rank: Current rank (0-indexed)
        world_size: Total number of ranks
        mode: Distribution mode
            - "interleave": Round-robin (0,4,8,... for rank 0 with 4 GPUs)
            - "contiguous": Contiguous blocks (0-499 for rank 0 with 2000 items, 4 GPUs)

    Returns:
        List of items assigned to this rank
    """
    if world_size <= 1:
        return list(items)

    if mode == "contiguous":
        # Contiguous block assignment
        n = len(items)
        base_size = n // world_size
        remainder = n % world_size
        # Ranks 0..remainder-1 get one extra item
        if rank < remainder:
            start = rank * (base_size + 1)
            end = start + base_size + 1
        else:
            start = remainder * (base_size + 1) + (rank - remainder) * base_size
            end = start + base_size
        return list(items[start:end])
    else:
        # Interleave (round-robin) - default
        return [item for idx, item in enumerate(items) if idx % world_size == rank]
