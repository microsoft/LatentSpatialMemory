from __future__ import annotations

from typing import Iterable, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_process.geometry import (
    transform_points,
    unproject_depth_to_points,
    voxel_indices,
)
from data_process.types import VideoGeometry


def save_point_cloud_ply(
    points: np.ndarray,
    out_path: str,
    colors: Optional[np.ndarray] = None,
    trajectory_points: Optional[np.ndarray] = None,
    trajectory_color: tuple[int, int, int] = (255, 0, 0),
) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if colors is not None:
        if colors.shape[0] != points.shape[0] or colors.shape[1] != 3:
            raise ValueError("colors must be Nx3 and match points")
    if trajectory_points is not None:
        if trajectory_points.ndim != 2 or trajectory_points.shape[1] != 3:
            raise ValueError("trajectory_points must be Nx3")

    has_colors = colors is not None
    if has_colors and trajectory_points is not None:
        traj_colors = np.tile(
            np.array(trajectory_color, dtype=np.uint8), (trajectory_points.shape[0], 1)
        )
        colors_all = np.concatenate([colors.astype(np.uint8), traj_colors], axis=0)
    elif has_colors:
        colors_all = colors.astype(np.uint8)
    else:
        colors_all = None

    vertex_count = points.shape[0] + (
        trajectory_points.shape[0] if trajectory_points is not None else 0
    )
    edge_count = 0
    if trajectory_points is not None and trajectory_points.shape[0] > 1:
        edge_count = trajectory_points.shape[0] - 1

    with open(out_path, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertex_count}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors_all is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        if edge_count > 0:
            f.write(f"element edge {edge_count}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
        f.write("end_header\n")

        if colors_all is None:
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            if trajectory_points is not None:
                for p in trajectory_points:
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            colors_u8 = np.clip(colors_all, 0, 255).astype(np.uint8)
            for p, c in zip(points, colors_u8[: points.shape[0]]):
                f.write(
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
                )
            if trajectory_points is not None:
                offset = points.shape[0]
                for p, c in zip(trajectory_points, colors_u8[offset:]):
                    f.write(
                        f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
                    )

        if edge_count > 0:
            base = points.shape[0]
            for i in range(edge_count):
                v1 = base + i
                v2 = base + i + 1
                f.write(f"{v1} {v2}\n")


def sample_points(
    points: np.ndarray,
    max_points: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def set_axes_equal(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = maxs - mins
    max_span = float(np.max(spans))
    if max_span <= 0:
        max_span = 1.0
    centers = (mins + maxs) / 2.0
    half = max_span / 2.0
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)


def collect_scene_points(
    geometry: VideoGeometry,
    indices: Iterable[int],
    voxel_size: float,
    dynamic_masks: Optional[np.ndarray] = None,
) -> np.ndarray:
    points_all = []
    for idx in indices:
        depth = geometry.depths[idx]
        K = geometry.intrinsics[idx]
        c2w = geometry.poses_c2w[idx]
        mask = depth > 0
        if geometry.masks is not None:
            mask = mask & geometry.masks[idx]
        if dynamic_masks is not None:
            mask = mask & (~dynamic_masks[idx])

        points_cam = unproject_depth_to_points(depth, K, mask=mask)
        if points_cam.size == 0:
            continue
        points_world = transform_points(points_cam, c2w)
        points_all.append(points_world)

    if not points_all:
        return np.zeros((0, 3), dtype=np.float32)

    points = np.concatenate(points_all, axis=0)
    if voxel_size > 0:
        vox = voxel_indices(points, voxel_size)
        _, unique_idx = np.unique(vox, axis=0, return_index=True)
        points = points[unique_idx]
    return points.astype(np.float32)


def plot_point_cloud(
    points: np.ndarray,
    out_path: str,
    max_points: int = 200_000,
    seed: Optional[int] = None,
    elev: float = 20.0,
    azim: float = -60.0,
    title: Optional[str] = None,
) -> None:
    if points.size == 0:
        raise ValueError("Point cloud is empty.")

    pts = sample_points(points, max_points=max_points, seed=seed)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=0.5,
        c=pts[:, 2],
        cmap="viridis",
        alpha=0.8,
        linewidths=0,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)
    set_axes_equal(ax, pts)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_camera_trajectory(
    poses_c2w: np.ndarray,
    out_path: str,
    stride: int = 1,
    axis_stride: int = 10,
    axis_scale: float = 0.1,
    show_axes: bool = True,
    elev: float = 20.0,
    azim: float = -60.0,
    title: Optional[str] = None,
) -> None:
    if poses_c2w.ndim != 3 or poses_c2w.shape[1:] != (4, 4):
        raise ValueError("poses_c2w must be (N,4,4)")

    indices = np.arange(0, poses_c2w.shape[0], stride)
    poses = poses_c2w[indices]
    centers = poses[:, :3, 3]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="tab:red", linewidth=1.0)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color="tab:red", s=4)

    if show_axes:
        axis_indices = indices[::axis_stride] if axis_stride > 0 else indices
        axis_poses = poses_c2w[axis_indices]
        for pose in axis_poses:
            origin = pose[:3, 3]
            rot = pose[:3, :3]
            dirs = [rot[:, 0], rot[:, 1], rot[:, 2]]
            colors = ["r", "g", "b"]
            for d, c in zip(dirs, colors):
                ax.quiver(
                    origin[0],
                    origin[1],
                    origin[2],
                    d[0],
                    d[1],
                    d[2],
                    length=axis_scale,
                    normalize=True,
                    color=c,
                    linewidth=0.8,
                )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)
    set_axes_equal(ax, centers)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_point_cloud_with_trajectory(
    points: np.ndarray,
    poses_c2w: np.ndarray,
    out_path: str,
    max_points: int = 200_000,
    seed: Optional[int] = None,
    traj_stride: int = 1,
    axis_stride: int = 10,
    axis_scale: float = 0.1,
    show_axes: bool = True,
    elev: float = 20.0,
    azim: float = -60.0,
    title: Optional[str] = None,
) -> None:
    if points.size == 0:
        raise ValueError("Point cloud is empty.")
    if poses_c2w.ndim != 3 or poses_c2w.shape[1:] != (4, 4):
        raise ValueError("poses_c2w must be (N,4,4)")

    pts = sample_points(points, max_points=max_points, seed=seed)

    indices = np.arange(0, poses_c2w.shape[0], traj_stride)
    poses = poses_c2w[indices]
    centers = poses[:, :3, 3]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=0.5,
        c=pts[:, 2],
        cmap="viridis",
        alpha=0.6,
        linewidths=0,
    )
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="tab:red", linewidth=1.0)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color="tab:red", s=4)

    if show_axes:
        axis_indices = indices[::axis_stride] if axis_stride > 0 else indices
        axis_poses = poses_c2w[axis_indices]
        for pose in axis_poses:
            origin = pose[:3, 3]
            rot = pose[:3, :3]
            dirs = [rot[:, 0], rot[:, 1], rot[:, 2]]
            colors = ["r", "g", "b"]
            for d, c in zip(dirs, colors):
                ax.quiver(
                    origin[0],
                    origin[1],
                    origin[2],
                    d[0],
                    d[1],
                    d[2],
                    length=axis_scale,
                    normalize=True,
                    color=c,
                    linewidth=0.8,
                )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)
    set_axes_equal(ax, np.vstack([pts, centers]))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
