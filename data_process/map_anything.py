"""
MapAnything 几何估计模块

使用MapAnything模型从视频帧估计:
- 深度图 (depth_z)
- 相机位姿 (camera_poses, c2w)
- 相机内参 (intrinsics)

支持分chunk处理长视频，并通过Umeyama对齐保证chunk间位姿一致性
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from data_process._0_0_0_root_assign import MapAnythingConfig
from data_process.types import VideoGeometry


@dataclass
class ChunkResult:
    """单个chunk的推理结果"""

    frames: np.ndarray
    depths: np.ndarray
    masks: Optional[np.ndarray]
    intrinsics: np.ndarray
    poses_c2w: np.ndarray


def _to_numpy(tensor):
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def umeyama_alignment(X: np.ndarray, Y: np.ndarray, with_scale: bool = True):
    """Umeyama对齐算法: 计算从X到Y的相似变换(旋转+平移+缩放)"""
    if X.shape != Y.shape:
        raise ValueError("Alignment expects same shape inputs.")
    n = X.shape[0]
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    cov = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    if with_scale:
        var_x = np.sum(Xc**2) / n
        scale = np.sum(D * np.diag(S)) / (var_x + 1e-8)
    else:
        scale = 1.0
    t = mu_y - scale * R @ mu_x
    return R, t, scale


def build_similarity_transform(R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = (s * R).astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


class MapAnythingEstimator:
    """MapAnything模型封装，用于视频几何估计"""

    def __init__(self, config: MapAnythingConfig):
        """加载MapAnything模型"""
        self.config = config
        device = torch.device(config.device)
        if config.device != "cpu" and not torch.cuda.is_available():
            device = torch.device("cpu")

        from mapanything.models import MapAnything
        from mapanything.utils.image import preprocess_inputs

        self._preprocess_inputs = preprocess_inputs
        self.device = device
        self.model = MapAnything.from_pretrained(config.model_id).to(device)
        encoder = getattr(self.model, "encoder", None)
        self.norm_type = getattr(encoder, "data_norm_type", "dinov2")

    def _infer_chunk(self, frames: List[np.ndarray]) -> ChunkResult:
        """对一个chunk的帧进行推理，返回深度、位姿、内参"""
        views = [{"img": frame} for frame in frames]
        processed_views = self._preprocess_inputs(
            views,
            resize_mode=self.config.resize_mode,
            size=self.config.size,
            norm_type=self.norm_type,
            resolution_set=self.config.resolution_set,
        )

        with torch.no_grad():
            preds = self.model.infer(
                processed_views,
                memory_efficient_inference=self.config.memory_efficient_inference,
                use_amp=self.config.use_amp,
                amp_dtype=self.config.amp_dtype,
                apply_mask=self.config.apply_mask,
                mask_edges=self.config.mask_edges,
                apply_confidence_mask=self.config.apply_confidence_mask,
                confidence_percentile=self.config.confidence_percentile,
            )

        frames_out = []
        depths = []
        masks = []
        intrinsics = []
        poses = []

        for pred in preds:
            img_no_norm = _to_numpy(pred.get("img_no_norm"))
            if img_no_norm is None:
                raise RuntimeError("MapAnything did not return img_no_norm.")
            img_no_norm = img_no_norm.squeeze(0)
            if img_no_norm.max() <= 1.0:
                img_no_norm = (img_no_norm * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_no_norm = img_no_norm.clip(0, 255).astype(np.uint8)
            frames_out.append(img_no_norm)

            depth = _to_numpy(pred.get("depth_z"))
            if depth is None:
                raise RuntimeError("MapAnything did not return depth_z.")
            depth = depth.squeeze(0).squeeze(-1).astype(np.float32)
            depths.append(depth)

            mask = pred.get("mask")
            if mask is not None:
                mask = _to_numpy(mask).squeeze(0).squeeze(-1).astype(bool)
            masks.append(mask)

            K = _to_numpy(pred.get("intrinsics"))
            if K is None:
                raise RuntimeError("MapAnything did not return intrinsics.")
            intrinsics.append(K.squeeze(0).astype(np.float32))

            pose = _to_numpy(pred.get("camera_poses"))
            if pose is None:
                raise RuntimeError("MapAnything did not return camera_poses.")
            poses.append(pose.squeeze(0).astype(np.float32))

        frames_arr = np.stack(frames_out, axis=0)
        depths_arr = np.stack(depths, axis=0)
        intrinsics_arr = np.stack(intrinsics, axis=0)
        poses_arr = np.stack(poses, axis=0)
        masks_arr = None
        if any(m is not None for m in masks):
            masks_arr = np.stack(
                [
                    m if m is not None else np.ones_like(depths_arr[0], dtype=bool)
                    for m in masks
                ],
                axis=0,
            )

        return ChunkResult(
            frames=frames_arr,
            depths=depths_arr,
            masks=masks_arr,
            intrinsics=intrinsics_arr,
            poses_c2w=poses_arr,
        )

    def estimate_video_geometry(
        self,
        frames: List[np.ndarray],
        frame_indices: Optional[List[int]] = None,
    ) -> VideoGeometry:
        """
        估计视频的几何信息

        对于长视频，分chunk处理并通过overlap区域进行Umeyama对齐，
        确保整个视频的位姿在同一坐标系下
        """
        if not frames:
            raise ValueError("frames is empty")

        chunk_size = self.config.chunk_size or len(frames)
        overlap = int(self.config.chunk_overlap)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >=0 and < chunk_size")

        original_h, original_w = frames[0].shape[:2]

        if chunk_size >= len(frames):
            result = self._infer_chunk(frames)
            processed_h, processed_w = result.frames.shape[1:3]
            return VideoGeometry(
                frames=result.frames,
                depths=result.depths,
                intrinsics=result.intrinsics,
                poses_c2w=result.poses_c2w,
                masks=result.masks,
                frame_indices=np.array(
                    frame_indices if frame_indices else list(range(len(frames))),
                    dtype=np.int32,
                ),
                original_size=(original_h, original_w),
                processed_size=(processed_h, processed_w),
            )

        step = chunk_size - overlap
        if step <= 0:
            raise ValueError("chunk_size must be greater than chunk_overlap")

        all_frames = []
        all_depths = []
        all_intrinsics = []
        all_poses = []
        all_masks = []

        start = 0
        processed_size = None
        global_indices = frame_indices if frame_indices else list(range(len(frames)))

        while start < len(frames):
            end = min(start + chunk_size, len(frames))
            chunk_frames = frames[start:end]
            result = self._infer_chunk(chunk_frames)

            if processed_size is None:
                processed_size = result.frames.shape[1:3]
            elif processed_size != result.frames.shape[1:3]:
                raise RuntimeError(
                    "Chunk outputs have inconsistent sizes; adjust resize settings."
                )

            if start > 0 and overlap > 0:
                prev_centers = np.array(
                    [c[:3, 3] for c in all_poses[-overlap:]], dtype=np.float32
                )
                curr_centers = np.array(
                    [c[:3, 3] for c in result.poses_c2w[:overlap]], dtype=np.float32
                )
                if prev_centers.shape[0] >= 3:
                    R, t, s = umeyama_alignment(
                        curr_centers, prev_centers, with_scale=True
                    )
                else:
                    R = np.eye(3, dtype=np.float32)
                    s = 1.0
                    t = (
                        prev_centers[0] - curr_centers[0]
                        if prev_centers.size
                        else np.zeros(3, dtype=np.float32)
                    )
                T_align = build_similarity_transform(R, t, s)
                result.poses_c2w = T_align @ result.poses_c2w

            if start == 0:
                all_frames.extend(list(result.frames))
                all_depths.extend(list(result.depths))
                all_intrinsics.extend(list(result.intrinsics))
                all_poses.extend(list(result.poses_c2w))
                if result.masks is not None:
                    all_masks.extend(list(result.masks))
            else:
                trim = overlap
                all_frames.extend(list(result.frames[trim:]))
                all_depths.extend(list(result.depths[trim:]))
                all_intrinsics.extend(list(result.intrinsics[trim:]))
                all_poses.extend(list(result.poses_c2w[trim:]))
                if result.masks is not None:
                    all_masks.extend(list(result.masks[trim:]))

            start += step

        frames_arr = np.stack(all_frames, axis=0)
        depths_arr = np.stack(all_depths, axis=0)
        intrinsics_arr = np.stack(all_intrinsics, axis=0)
        poses_arr = np.stack(all_poses, axis=0)
        masks_arr = None
        if all_masks:
            masks_arr = np.stack(all_masks, axis=0)

        processed_h, processed_w = frames_arr.shape[1:3]
        return VideoGeometry(
            frames=frames_arr,
            depths=depths_arr,
            intrinsics=intrinsics_arr,
            poses_c2w=poses_arr,
            masks=masks_arr,
            frame_indices=np.array(global_indices, dtype=np.int32),
            original_size=(original_h, original_w),
            processed_size=(processed_h, processed_w),
        )
