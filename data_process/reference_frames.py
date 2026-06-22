"""
参考帧选择模块

基于空间IOU选择参考帧:
1. 将每帧深度图反投影为voxel occupancy
2. 计算candidate帧与target帧的voxel IOU
3. 选择IOU超过阈值的帧作为参考帧
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from data_process.geometry import (
    transform_points,
    unproject_depth_to_points,
    voxel_indices,
)


def occupancy_from_frame(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    voxel_size: float,
    valid_mask: Optional[np.ndarray] = None,
    dynamic_mask: Optional[np.ndarray] = None,
) -> set[tuple[int, int, int]]:
    """从单帧深度图计算voxel occupancy集合"""
    mask = depth > 0
    if valid_mask is not None:
        mask = mask & valid_mask
    if dynamic_mask is not None:
        mask = mask & (~dynamic_mask)

    points_cam = unproject_depth_to_points(depth, K, mask=mask)
    if points_cam.size == 0:
        return set()

    points_world = transform_points(points_cam, c2w)
    vox = voxel_indices(points_world, voxel_size)
    return set(map(tuple, vox.tolist()))


def iou_occupancy(a: set[tuple[int, int, int]], b: set[tuple[int, int, int]]) -> float:
    """计算两个voxel集合的IOU (交集/并集)"""
    if not a and not b:
        return 0.0
    inter = a.intersection(b)
    union = a.union(b)
    if not union:
        return 0.0
    return float(len(inter)) / float(len(union))


class RefSelectionResult:
    """Result of reference frame selection with diagnostic info."""

    def __init__(
        self,
        indices: list[int],
        ious: list[float],
        stats: dict,
    ):
        self.indices = indices
        self.ious = ious
        self.stats = stats

    @property
    def count(self) -> int:
        return len(self.indices)

    def get_status_str(self) -> str:
        """Get a human-readable status string for logging."""
        if self.count > 0:
            return f"ref={self.count}, best_iou={self.stats['best_iou']:.3f}"
        else:
            reason = self.stats.get("no_ref_reason", "unknown")
            best = self.stats.get("best_iou", 0)
            thresh = self.stats.get("threshold", 0)
            if reason == "max_refs_zero":
                return "ref=0 (max_refs=0)"
            elif reason == "no_candidates":
                return "ref=0 (no candidates)"
            elif reason == "iou_below_threshold":
                return f"ref=0 (best_iou={best:.3f}<{thresh:.3f})"
            else:
                return f"ref=0 ({reason})"


def select_reference_frames(
    candidate_indices: Iterable[int],
    target_indices: Iterable[int],
    depths: np.ndarray,
    intrinsics: np.ndarray,
    poses_c2w: np.ndarray,
    voxel_size: float,
    stride: int,
    iou_threshold: float,
    max_refs: int,
    valid_masks: Optional[np.ndarray] = None,
    dynamic_masks: Optional[np.ndarray] = None,
    return_result: bool = False,
):
    """
    Select reference frames based on spatial overlap with target frames.

    Following Spatia paper (Algorithm 1): For each target frame, find the candidate
    with highest IOU. A candidate is selected as reference if its max IOU with any
    target frame exceeds the threshold.

    This is different from merging all target frames - we compute per-frame IOU
    which gives much higher overlap scores.

    Args:
        return_result: If True, returns RefSelectionResult with diagnostic info.
                      If False (default), returns (indices, ious) for backward compatibility.
    """
    stats = {
        "threshold": iou_threshold,
        "max_refs": max_refs,
        "stride": stride,
        "voxel_size": voxel_size,
    }

    if max_refs <= 0:
        stats["no_ref_reason"] = "max_refs_zero"
        stats["best_iou"] = 0.0
        if return_result:
            return RefSelectionResult([], [], stats)
        return [], []

    target_list = list(target_indices)
    candidates = list(candidate_indices)
    if stride > 1:
        candidates = candidates[::stride]

    stats["num_targets"] = len(target_list)
    stats["num_candidates"] = len(candidates)

    if not candidates:
        stats["no_ref_reason"] = "no_candidates"
        stats["best_iou"] = 0.0
        if return_result:
            return RefSelectionResult([], [], stats)
        return [], []

    # Pre-compute occupancy for all target frames
    target_occs = []
    for idx in target_list:
        occ = occupancy_from_frame(
            depth=depths[idx],
            K=intrinsics[idx],
            c2w=poses_c2w[idx],
            voxel_size=voxel_size,
            valid_mask=None if valid_masks is None else valid_masks[idx],
            dynamic_mask=None if dynamic_masks is None else dynamic_masks[idx],
        )
        target_occs.append(occ)

    # Pre-compute occupancy for all candidate frames
    candidate_occs = {}
    for idx in candidates:
        occ = occupancy_from_frame(
            depth=depths[idx],
            K=intrinsics[idx],
            c2w=poses_c2w[idx],
            voxel_size=voxel_size,
            valid_mask=None if valid_masks is None else valid_masks[idx],
            dynamic_mask=None if dynamic_masks is None else dynamic_masks[idx],
        )
        candidate_occs[idx] = occ

    # For each candidate, compute max IOU across all target frames
    # (Following Spatia paper: select candidate with highest spatial overlap)
    scored = []
    all_ious = []  # For debugging
    for c_idx in candidates:
        c_occ = candidate_occs[c_idx]
        max_iou = 0.0
        for t_occ in target_occs:
            iou = iou_occupancy(c_occ, t_occ)
            if iou > max_iou:
                max_iou = iou
        all_ious.append((c_idx, max_iou, len(c_occ)))
        if max_iou >= iou_threshold:
            scored.append((c_idx, max_iou))

    # Compute statistics
    best_iou = max(x[1] for x in all_ious) if all_ious else 0.0
    avg_iou = sum(x[1] for x in all_ious) / len(all_ious) if all_ious else 0.0
    stats["best_iou"] = best_iou
    stats["avg_iou"] = avg_iou

    if len(scored) == 0:
        stats["no_ref_reason"] = "iou_below_threshold"

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = scored[:max_refs]
    indices = [s[0] for s in selected]
    ious = [s[1] for s in selected]

    if return_result:
        return RefSelectionResult(indices, ious, stats)
    return indices, ious
