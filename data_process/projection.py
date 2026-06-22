"""
点云投影模块

将3D点云投影到2D图像平面，生成:
- depth: 深度图
- mask: 有效像素mask
- rgb: 颜色投影 (可选)

使用z-buffer处理遮挡关系 (保留最近的点)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from data_process.geometry import project_points, transform_points


def z_buffer_projection(
    points_world: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    image_size: tuple[int, int],
):
    """基础z-buffer投影: 世界坐标点 -> 深度图和mask"""
    channels = list(channels)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError("points_world must be Nx3")

    w2c = np.linalg.inv(c2w)
    points_cam = transform_points(points_world, w2c)
    uv, z = project_points(points_cam, K)

    H, W = image_size
    # Filter out invalid uv values (NaN/inf) and out-of-range before casting to int
    int32_max = np.iinfo(np.int32).max
    uv_valid = (
        np.isfinite(uv).all(axis=1)
        & (z > 0)
        & (np.abs(uv[:, 0]) < int32_max)
        & (np.abs(uv[:, 1]) < int32_max)
    )
    u = np.zeros(len(uv), dtype=np.int32)
    v = np.zeros(len(uv), dtype=np.int32)
    if uv_valid.any():
        u[uv_valid] = np.rint(uv[uv_valid, 0]).astype(np.int32)
        v[uv_valid] = np.rint(uv[uv_valid, 1]).astype(np.int32)
    valid = uv_valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    depth = np.full((H, W), np.inf, dtype=np.float32)
    if valid.any():
        flat = v[valid] * W + u[valid]
        np.minimum.at(depth.ravel(), flat, z[valid])
    mask = np.isfinite(depth)
    depth[~mask] = 0.0
    return depth, mask


def render_projection(
    points_world: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    image_size: tuple[int, int],
    channels: Iterable[str],
    colors: np.ndarray | None = None,
    fill_holes_kernel: int = 0,
) -> np.ndarray:
    """
    渲染点云投影

    将世界坐标点云投影到指定相机视角，支持:
    - depth: 深度通道
    - mask: 有效区域mask
    - rgb: 颜色通道 (需提供colors)

    可选择对投影结果进行孔洞填充 (fill_holes_kernel)
    """
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError("points_world must be Nx3")

    w2c = np.linalg.inv(c2w)
    points_cam = transform_points(points_world, w2c)
    uv, z = project_points(points_cam, K)

    H, W = image_size
    # Filter out invalid uv values (NaN/inf) and out-of-range before casting to int
    int32_max = np.iinfo(np.int32).max
    uv_valid = (
        np.isfinite(uv).all(axis=1)
        & (z > 0)
        & (np.abs(uv[:, 0]) < int32_max)
        & (np.abs(uv[:, 1]) < int32_max)
    )
    u = np.zeros(len(uv), dtype=np.int32)
    v = np.zeros(len(uv), dtype=np.int32)
    if uv_valid.any():
        u[uv_valid] = np.rint(uv[uv_valid, 0]).astype(np.int32)
        v[uv_valid] = np.rint(uv[uv_valid, 1]).astype(np.int32)
    valid = uv_valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    depth_img = np.zeros((H, W), dtype=np.float32)
    mask_img = np.zeros((H, W), dtype=bool)
    rgb_img = np.zeros((H, W, 3), dtype=np.float32)

    if valid.any():
        flat = v[valid] * W + u[valid]
        depth_valid = z[valid]
        order = np.argsort(depth_valid)
        flat_sorted = flat[order]
        depth_sorted = depth_valid[order]
        unique_flat, first_idx = np.unique(flat_sorted, return_index=True)

        depth_min = np.full(H * W, np.inf, dtype=np.float32)
        depth_min[unique_flat] = depth_sorted[first_idx]
        mask_img = np.isfinite(depth_min).reshape(H, W)
        depth_img = depth_min.reshape(H, W)
        depth_img[~mask_img] = 0.0

        if "rgb" in channels:
            if colors is None:
                raise ValueError(
                    "colors must be provided when requesting rgb projection"
                )
            colors_valid = colors[valid]
            colors_sorted = colors_valid[order]
            rgb_flat = np.zeros((H * W, 3), dtype=np.float32)
            rgb_flat[unique_flat] = colors_sorted[first_idx].astype(np.float32)
            rgb_img = rgb_flat.reshape(H, W, 3)

    if fill_holes_kernel > 0 and (mask_img.any()):
        import cv2

        kernel_size = max(1, int(fill_holes_kernel))

        # 策略: 对所有未覆盖的像素进行inpaint，而不仅仅是形态学闭运算检测到的小孔
        # 这样可以填充点云稀疏导致的所有空洞
        if "rgb" in channels:
            rgb_u8 = np.clip(rgb_img, 0, 255).astype(np.uint8)
            # inpaint_mask: 所有没有点云覆盖的像素
            inpaint_mask = (~mask_img).astype(np.uint8) * 255
            # 使用较大的inpaint半径以填充较大空洞
            radius = max(3, kernel_size)
            rgb_u8 = cv2.inpaint(rgb_u8, inpaint_mask, radius, cv2.INPAINT_TELEA)
            rgb_img = rgb_u8.astype(np.float32)

        # 更新mask为全图（因为inpaint后所有像素都有值）
        mask_img = np.ones((H, W), dtype=bool)

    parts: list[np.ndarray] = []
    for ch in channels:
        if ch == "depth":
            parts.append(depth_img[..., None])
        elif ch == "mask":
            parts.append(mask_img.astype(np.float32)[..., None])
        elif ch == "rgb":
            parts.append(rgb_img.astype(np.float32))
        else:
            raise ValueError(f"Unsupported projection channel: {ch}")

    if not parts:
        return np.zeros((H, W, 0), dtype=np.float32)
    return np.concatenate(parts, axis=-1).astype(np.float32)
