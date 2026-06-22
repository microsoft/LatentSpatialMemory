"""
帧索引采样模块

从视频中采样训练样本的帧索引:
- P (Preceding): 前置帧，用于初始化
- T (Target): 目标帧，模型需要生成的帧
- C (Candidate): 候选帧，用于选择参考帧R
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from data_process.types import SampleIndices


def sample_frame_indices(
    num_frames: int,
    N_target: int,
    M_pre: int,
    min_gap_for_candidates: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> SampleIndices:
    """
    采样帧索引

    随机选择起始点t0，然后:
    - preceding: [t0-M_pre, t0) 共M_pre帧
    - target: [t0, t0+N_target) 共N_target帧
    - candidate: 剩余所有帧 (排除P和T)

    示例 (81帧视频, M_pre=8, N_target=33):
      若t0=40，则:
      - P: [32, 33, ..., 39] (8帧)
      - T: [40, 41, ..., 72] (33帧)
      - C: [0, ..., 31] + [73, ..., 80] (40帧)
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if N_target <= 0 or M_pre <= 0:
        raise ValueError("N_target and M_pre must be positive")
    if num_frames < N_target + M_pre:
        raise ValueError("Not enough frames to sample training sample")

    if rng is None:
        rng = np.random.default_rng()

    t0_min = M_pre
    t0_max = num_frames - N_target
    if t0_min > t0_max:
        raise ValueError("Invalid sampling range for t0")

    t0 = int(rng.integers(t0_min, t0_max + 1))
    target_indices = list(range(t0, t0 + N_target))
    preceding_indices = list(range(t0 - M_pre, t0))

    preceding_set = set(preceding_indices)
    target_set = set(target_indices)
    candidate_indices = [
        i for i in range(num_frames) if i not in preceding_set and i not in target_set
    ]

    if min_gap_for_candidates > 0:
        protected = sorted(preceding_indices + target_indices)

        def far_enough(idx: int) -> bool:
            return min(abs(idx - t) for t in protected) >= min_gap_for_candidates

        candidate_indices = [i for i in candidate_indices if far_enough(i)]

    return SampleIndices(
        t0=t0,
        preceding_indices=preceding_indices,
        target_indices=target_indices,
        candidate_indices=candidate_indices,
    )


# Backward compatibility alias
sample_episode_indices = sample_frame_indices
