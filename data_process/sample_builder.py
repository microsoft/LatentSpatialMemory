"""
训练样本构建模块

核心功能:
1. 从视频几何信息中采样帧索引 (preceding/target/reference)
2. 使用 target 的第一帧构建 latent scene point cloud (排除动态物体)
3. 将 latent point cloud 投影到各帧生成scene projection
4. 选择参考帧 (基于空间IOU)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

from data_process.data_config import SampleConfig
from data_process.reference_frames import (
    RefSelectionResult,
    select_reference_frames,
)
from data_process.sample_indices import sample_frame_indices
from data_process.types import SampleIndices, VideoGeometry
from mirage.latent_point_cloud import LatentPointCloud


def resize_frames(frames: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """将帧序列resize到目标尺寸 (H, W)"""
    target_h, target_w = target_size
    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames
    resized = []
    for frame in frames:
        resized.append(
            cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        )
    return np.stack(resized, axis=0)


def scale_intrinsics(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """缩放相机内参矩阵以适应不同分辨率"""
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    return K_scaled


def build_scene_exclusion_mask(
    geometry: VideoGeometry,
    dynamic_masks: Optional[np.ndarray],
    scene_idx: int,
) -> Optional[np.ndarray]:
    """Build a single exclusion mask for latent point cloud construction."""
    exclusion_mask = None
    if geometry.masks is not None:
        exclusion_mask = ~geometry.masks[scene_idx]
    if dynamic_masks is not None:
        dynamic_mask = dynamic_masks[scene_idx]
        exclusion_mask = (
            dynamic_mask.copy()
            if exclusion_mask is None
            else (exclusion_mask | dynamic_mask)
        )
    return exclusion_mask


def build_latent_projection(
    latent_point_cloud: LatentPointCloud,
    geometry: VideoGeometry,
    frame_indices: list[int],
    temporal_stride: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Project a latent point cloud to temporally sampled target frames."""
    sampled_frame_indices = list(frame_indices)[::temporal_stride]
    projections, masks = _project_latent_frames_to_numpy(
        latent_point_cloud=latent_point_cloud,
        geometry=geometry,
        frame_indices=sampled_frame_indices,
    )
    return projections, masks, sampled_frame_indices


def _project_latent_frames_to_numpy(
    latent_point_cloud: LatentPointCloud,
    geometry: VideoGeometry,
    frame_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project latent point cloud frames with one CPU sync per output tensor."""
    if not frame_indices:
        raise ValueError("frame_indices must contain at least one frame")

    H, W = geometry.frames.shape[1:3]
    latent_h, latent_w = latent_point_cloud.latent_hw
    scale_x = latent_w / W
    scale_y = latent_h / H

    projections = []
    masks = []
    for idx in frame_indices:
        intrinsics_latent = scale_intrinsics(geometry.intrinsics[idx], scale_x, scale_y)
        proj, proj_mask = latent_point_cloud.project(
            cam2world=geometry.poses_c2w[idx],
            intrinsics=intrinsics_latent,
        )
        projections.append(proj)
        masks.append(proj_mask)

    projection_array = torch.stack(projections, dim=0).detach().cpu().numpy()
    projection_array = projection_array.astype(np.float32, copy=False)
    mask_array = torch.stack(masks, dim=0).detach().cpu().numpy()
    return projection_array, mask_array


def build_training_sample(
    geometry: VideoGeometry,
    config: SampleConfig,
    dynamic_masks: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    original_frames: Optional[np.ndarray] = None,
    output_size: Optional[tuple[int, int]] = None,
    indices: Optional[SampleIndices] = None,
    scene_latent: Optional[torch.Tensor] = None,
    latent_projection_stride: int = 4,
):
    """
    从视频几何信息构建训练样本

    流程:
    1. sample_frame_indices: 随机采样t0，确定P(preceding)/T(target)/C(candidate)帧索引
    2. LatentPointCloud.from_video_geometry: 使用 target 第一帧 t0 构建 latent scene
    3. 将 latent point cloud 投影到P和T帧，生成scene projection
    4. select_reference_frames: 基于空间IOU选择参考帧R

    输出字典包含:
    - P_rgb/T_rgb/R_rgb: 前置帧/目标帧/参考帧的RGB
    - proj_P/proj_T: latent point cloud投影到P/T帧的结果
    - meta: 帧索引、参考帧选择统计等元信息
    """
    assert scene_latent is not None, (
        "scene_latent must be provided for latent scene projection"
    )

    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    if indices is None:
        num_frames = geometry.frames.shape[0]
        indices = sample_frame_indices(
            num_frames=num_frames,
            N_target=config.N_target,
            M_pre=config.M_pre,
            min_gap_for_candidates=config.min_gap_for_candidates,
            rng=rng,
        )

    scene_idx = int(indices.t0)

    H, W = geometry.frames.shape[1:3]  # MapAnything处理后的分辨率

    preceding_proj_mask = None
    target_proj_mask = None
    exclusion_mask = build_scene_exclusion_mask(
        geometry=geometry,
        dynamic_masks=dynamic_masks,
        scene_idx=scene_idx,
    )
    latent_point_cloud = LatentPointCloud.from_video_geometry(
        geometry=geometry,
        frame_idx=scene_idx,
        latent=scene_latent,
        mask=exclusion_mask,
        device=scene_latent.device,
    )
    valid_scene_points = (
        latent_point_cloud.points_world[latent_point_cloud.valid_mask]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    assert valid_scene_points.size > 0, (
        "Latent scene projection has no valid points; check geometry."
    )
    preceding_proj_indices = list(indices.preceding_indices)[
        ::latent_projection_stride
    ]
    target_proj_indices = list(indices.target_indices)[::latent_projection_stride]
    all_projection_indices = preceding_proj_indices + target_proj_indices
    all_projections, all_projection_masks = _project_latent_frames_to_numpy(
        latent_point_cloud=latent_point_cloud,
        geometry=geometry,
        frame_indices=all_projection_indices,
    )
    preceding_count = len(preceding_proj_indices)
    preceding_proj = all_projections[:preceding_count]
    target_proj = all_projections[preceding_count:]
    preceding_proj_mask = all_projection_masks[:preceding_count]
    target_proj_mask = all_projection_masks[preceding_count:]

    # 选择参考帧: 计算candidate帧与target帧的空间IOU，选择重叠度高的帧作为参考
    # 使用较大的voxel_size来容忍深度噪声
    iou_voxel_size = config.ref_iou_voxel_size
    if iou_voxel_size is None:
        iou_voxel_size = config.scene_voxel_size * 10

    ref_result: RefSelectionResult = select_reference_frames(
        candidate_indices=indices.candidate_indices,
        target_indices=indices.target_indices,
        depths=geometry.depths,
        intrinsics=geometry.intrinsics,
        poses_c2w=geometry.poses_c2w,
        voxel_size=iou_voxel_size,
        stride=config.K_ref_stride,
        iou_threshold=config.eps_iou,
        max_refs=config.max_refs,
        valid_masks=geometry.masks,
        dynamic_masks=dynamic_masks,
        return_result=True,
    )
    reference_indices = ref_result.indices
    reference_ious = ref_result.ious

    # Determine output size and source frames
    # If original_frames provided, use those for RGB output; otherwise use geometry.frames
    if original_frames is not None:
        # Ensure src_frames is a numpy array (load_video_frames returns list)
        src_frames = np.asarray(original_frames)
    else:
        src_frames = geometry.frames

    # Determine target output size
    if output_size is not None:
        out_h, out_w = output_size
    elif geometry.original_size is not None:
        out_h, out_w = geometry.original_size
    else:
        out_h, out_w = H, W  # Fall back to processed size

    # Extract RGB frames from source (original resolution if available)
    preceding_rgb = src_frames[indices.preceding_indices]
    target_rgb = src_frames[indices.target_indices]
    if reference_indices:
        reference_rgb = src_frames[reference_indices]
    else:
        reference_rgb = np.zeros((0, out_h, out_w, 3), dtype=np.uint8)

    # Resize RGB frames to output size if needed
    preceding_rgb = resize_frames(preceding_rgb, (out_h, out_w))
    target_rgb = resize_frames(target_rgb, (out_h, out_w))
    if reference_rgb.shape[0] > 0:
        reference_rgb = resize_frames(reference_rgb, (out_h, out_w))

    sample = {
        # RGB frames
        "P_rgb": preceding_rgb,
        "T_rgb": target_rgb,
        "R_rgb": reference_rgb,
        # Camera parameters
        "P_poses_c2w": geometry.poses_c2w[indices.preceding_indices],
        "T_poses_c2w": geometry.poses_c2w[indices.target_indices],
        "P_intrinsics": geometry.intrinsics[indices.preceding_indices],
        "T_intrinsics": geometry.intrinsics[indices.target_indices],
        # Scene data
        "proj_P": preceding_proj,
        "proj_T": target_proj,
        # Metadata
        "meta": {
            "t0": indices.t0,
            "P_idx": indices.preceding_indices,
            "T_idx": indices.target_indices,
            "C_idx": indices.candidate_indices,
            "scene_idx": scene_idx,
            "R_idx": reference_indices,
            "R_iou": reference_ious,
            "R_stats": ref_result.stats,  # Contains best_iou, avg_iou, threshold, no_ref_reason (if applicable)
            "point_cloud_type": "latent",
            "projection_channels": ["latent"],
            "projection_layout": "tchw",
            "latent_projection_stride": latent_projection_stride,
            "proj_P_idx": preceding_proj_indices,
            "proj_T_idx": target_proj_indices,
            "output_size": (out_h, out_w),
        },
    }
    if preceding_proj_mask is not None:
        sample["proj_P_mask"] = preceding_proj_mask
    if target_proj_mask is not None:
        sample["proj_T_mask"] = target_proj_mask
    return sample
