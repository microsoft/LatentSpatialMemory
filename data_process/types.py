"""
Spatia 数据类型定义
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SampleIndices:
    """
    训练样本的帧索引

    t0: target帧的起始索引
    preceding_indices: 前置帧索引列表 (P)
    target_indices: 目标帧索引列表 (T)
    candidate_indices: 候选帧索引列表 (C)，用于选择参考帧
    """

    t0: int
    preceding_indices: list[int]
    target_indices: list[int]
    candidate_indices: list[int]


EpisodeIndices = SampleIndices  # 兼容别名


@dataclass
class VideoGeometry:
    """
    视频几何信息

    由MapAnything估计得到，包含:
    - frames: RGB帧 (L, H, W, 3)
    - depths: 深度图 (L, H, W)
    - intrinsics: 相机内参 (L, 3, 3)
    - poses_c2w: 相机位姿，camera-to-world (L, 4, 4)
    - masks: 有效区域mask (L, H, W)
    """

    frames: np.ndarray
    depths: np.ndarray
    intrinsics: np.ndarray
    poses_c2w: np.ndarray
    masks: Optional[np.ndarray] = None
    frame_indices: Optional[np.ndarray] = None
    original_size: Optional[tuple[int, int]] = None  # 原始分辨率 (H, W)
    processed_size: Optional[tuple[int, int]] = None  # MapAnything处理后的分辨率
