"""
点云构建模块

从深度图反投影构建3D点云，用于场景表示和投影
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from data_process.geometry import (
    transform_points,
    unproject_depth_to_points,
    voxel_indices,
)


def build_scene_point_cloud(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    dynamic_mask: Optional[np.ndarray] = None,
    voxel_size: float = 0.05,
):
    """
    从单帧深度图构建场景点云

    流程:
    1. 根据valid_mask和dynamic_mask过滤像素 (排除无效区域和动态物体)
    2. 使用内参K将深度图反投影为相机坐标系下的3D点
    3. 使用c2w变换到世界坐标系
    4. 使用voxel下采样去除冗余点

    返回: (点云xyz, 点云颜色rgb)
    """
    if valid_mask is not None and valid_mask.shape != depth.shape:
        raise ValueError("valid_mask must match depth shape")
    if dynamic_mask is not None and dynamic_mask.shape != depth.shape:
        raise ValueError("dynamic_mask must match depth shape")

    mask = depth > 0
    if valid_mask is not None:
        mask = mask & valid_mask
    if dynamic_mask is not None:
        mask = mask & (~dynamic_mask)

    points_cam = unproject_depth_to_points(depth, K, mask=mask)
    if points_cam.size == 0:
        empty_xyz = np.zeros((0, 3), dtype=np.float32)
        if rgb is None:
            return empty_xyz, None
        empty_rgb = np.zeros((0, 3), dtype=np.uint8)
        return empty_xyz, empty_rgb

    points_world = transform_points(points_cam, c2w)
    if voxel_size > 0:
        vox = voxel_indices(points_world, voxel_size)
        _, unique_idx = np.unique(vox, axis=0, return_index=True)
        points_world = points_world[unique_idx]
    else:
        unique_idx = None

    if rgb is None:
        return points_world, None

    ys, xs = np.nonzero(mask)
    colors = rgb[ys, xs]
    if unique_idx is not None:
        colors = colors[unique_idx]
    return points_world, colors
