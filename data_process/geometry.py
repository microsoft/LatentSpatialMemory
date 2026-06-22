"""
几何变换基础函数

提供深度图与3D点云之间的转换:
- unproject: 深度图 -> 相机坐标系3D点
- project: 相机坐标系3D点 -> 像素坐标
- transform: 3D点坐标变换 (如 c2w, w2c)
- voxel: 体素化相关操作
"""

from __future__ import annotations

from typing import Optional

import numpy as np


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
