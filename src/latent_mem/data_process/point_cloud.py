"""
点云构建模块

从深度图反投影构建3D点云，用于场景表示和投影
"""

from __future__ import annotations

from typing import Optional

import numpy as np


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


def unproject_depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    mask: Optional[np.ndarray] = None,
    return_pixels: bool = False,
):
    """
    深度图反投影为相机坐标系3D点

    使用相机内参K将2D像素+深度转换为3D点:
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth
    """
    if depth.ndim != 2:
        raise ValueError("depth must be HxW")
    if K.shape != (3, 3):
        raise ValueError("K must be 3x3")

    H, W = depth.shape
    if mask is None:
        mask = depth > 0
    else:
        mask = mask & (depth > 0)

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        if return_pixels:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
        return np.zeros((0, 3), dtype=np.float32)

    z = depth[ys, xs].astype(np.float32)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=1)

    if return_pixels:
        pixels = np.stack([xs.astype(np.int32), ys.astype(np.int32)], axis=1)
        return points, pixels
    return points


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """使用4x4变换矩阵变换3D点 (如c2w将相机坐标转为世界坐标)"""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points.astype(np.float32), ones], axis=1)
    transformed = (transform @ homo.T).T[:, :3]
    return transformed.astype(np.float32)


def project_points(points: np.ndarray, K: np.ndarray):
    """将相机坐标系3D点投影到像素坐标，返回(uv坐标, z深度)"""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if K.shape != (3, 3):
        raise ValueError("K must be 3x3")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # Avoid divide by zero: use np.where to handle z <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(z > 0, (x / z) * fx + cx, np.nan)
        v = np.where(z > 0, (y / z) * fy + cy, np.nan)
    return np.stack([u, v], axis=1), z


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """体素下采样: 每个voxel只保留一个点"""
    if voxel_size <= 0:
        return points.astype(np.float32)
    if points.size == 0:
        return points.astype(np.float32)

    vox = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(vox, axis=0, return_index=True)
    return points[unique_idx].astype(np.float32)


def voxel_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """计算每个点所属的voxel索引 (用于IOU计算)"""
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    return np.floor(points / voxel_size).astype(np.int32)
