#!/usr/bin/env python3
"""
Spatia downstream data processing pipeline.

Pipeline stages:
1. Consume pre-sampled clips under output_root
2. Load foreground masks from sample assets
3. Load geometry from sample assets
4. Training sample building
"""

from __future__ import annotations

import gc
import json
import logging
import shutil
import traceback
import zipfile
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import torch

# Import config first - it sets up environment variables before other imports
from data_process._0_0_0_root_assign import (
    CONFIG,
    SampleConfig,
)
from data_process.clip_sampling import (
    CLIP_FILENAME,
    check_clip_complete,
    load_source_video_path,
)
from data_process.dataset_writer import save_training_sample, save_video
from data_process.distributed import get_rank_info, shard_items
from data_process.naming import get_sample_naming
from data_process.sample_builder import build_training_sample
from data_process.types import SampleIndices, VideoGeometry
from data_process.video_io import (
    get_video_fps,
    get_video_frame_count,
    load_video_frames,
)

logger = logging.getLogger(__name__)

# ============================================================================
# 工具函数
# ============================================================================


def _cleanup_memory() -> None:
    """清理GPU和CPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _get_point_cloud_vae_dtype(dtype_name: str) -> torch.dtype:
    """Resolve the dtype used for latent point cloud VAE encoding."""
    assert dtype_name in {"fp32", "fp16", "bf16"}, (
        f"Unsupported VAE dtype: {dtype_name}"
    )
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    return torch.bfloat16


def _encode_scene_latent(
    frame: np.ndarray,
    vae,
    device: str,
) -> torch.Tensor:
    """Encode one scene frame into a latent tensor."""
    assert frame.ndim == 3 and frame.shape[2] == 3, (
        f"frame must have shape (H, W, 3), got {frame.shape}"
    )
    frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
    stride_h, stride_w = vae.vae_stride[1], vae.vae_stride[2]
    assert frame_h % stride_h == 0 and frame_w % stride_w == 0, (
        "Scene frame resolution must be divisible by the VAE spatial stride. "
        f"Got {(frame_h, frame_w)} with stride {(stride_h, stride_w)}."
    )

    image_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(device)
    image_tensor = image_tensor / 255.0
    video_tensor = image_tensor.mul(2.0).sub(1.0).unsqueeze(1)
    with torch.no_grad():
        return vae.encode([video_tensor])[0].float().to(device)


# ============================================================================
# Helpers
# ============================================================================


def _get_video_hw(video_path: str | Path) -> tuple[int, int] | None:
    """获取视频的高宽 (H, W)"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    h, w = (
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    cap.release()
    return (h, w) if h > 0 and w > 0 else None


def _list_zip_members(archive_path: Path) -> list[str]:
    """List non-directory archive members in stable order."""
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if name and not name.endswith("/")]
    return sorted(names)


def _decode_image_bytes(
    image_bytes: bytes, flags: int, archive_path: Path
) -> np.ndarray:
    """Decode an image payload from zip bytes."""
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    assert image is not None, f"Failed to decode image from {archive_path}"
    return image


def _load_masks_from_zip(
    mask_zip_path: Path,
    expected_frames: int,
    expected_hw: tuple[int, int],
) -> np.ndarray:
    """Load per-frame PNG masks from a zip archive."""
    member_names = _list_zip_members(mask_zip_path)
    assert len(member_names) == expected_frames, (
        f"mask.zip frame count mismatch: expected {expected_frames}, "
        f"got {len(member_names)} in {mask_zip_path}"
    )

    expected_h, expected_w = expected_hw
    masks = []
    with zipfile.ZipFile(mask_zip_path) as archive:
        for member_name in member_names:
            decoded = _decode_image_bytes(
                archive.read(member_name),
                cv2.IMREAD_UNCHANGED,
                mask_zip_path,
            )
            if decoded.ndim == 3:
                decoded = decoded[..., 0]
            assert decoded.shape == (expected_h, expected_w), (
                f"Mask shape mismatch in {mask_zip_path}: expected "
                f"{(expected_h, expected_w)}, got {decoded.shape}"
            )
            masks.append(decoded > 0)
    return np.stack(masks, axis=0)


def _load_depths_from_zip(depth_zip_path: Path, expected_frames: int) -> np.ndarray:
    """Load per-frame EXR depths from a zip archive."""
    member_names = _list_zip_members(depth_zip_path)
    assert len(member_names) == expected_frames, (
        f"depth.zip frame count mismatch: expected {expected_frames}, "
        f"got {len(member_names)} in {depth_zip_path}"
    )

    depth_frames: dict[int, np.ndarray] = {}
    valid_shape: tuple[int, int] | None = None
    with zipfile.ZipFile(depth_zip_path) as archive:
        for member_name in member_names:
            frame_idx = int(Path(member_name).stem)
            with archive.open(member_name) as exr_file:
                try:
                    exr = OpenEXR.InputFile(exr_file)
                except OSError:
                    assert valid_shape is not None, (
                        f"Failed to decode first EXR frame in {depth_zip_path}; "
                        "cannot infer fallback depth shape."
                    )
                    logger.warning(
                        "Failed to load EXR file %s-%s. Returning an all-NaN depth map.",
                        depth_zip_path,
                        member_name,
                    )
                    depth_frames[frame_idx] = np.full(
                        valid_shape,
                        np.nan,
                        dtype=np.float32,
                    )
                    continue

                header = exr.header()
                data_window = header["dataWindow"]
                width = data_window.max.x - data_window.min.x + 1
                height = data_window.max.y - data_window.min.y + 1
                valid_shape = (height, width)

                channels = exr.channels(["Z"])
                depth_data = np.frombuffer(
                    channels[0],
                    dtype=np.float16,
                ).reshape(valid_shape)
                depth_frames[frame_idx] = depth_data.astype(np.float32, copy=True)

    expected_indices = list(range(expected_frames))
    actual_indices = sorted(depth_frames)
    assert actual_indices == expected_indices, (
        f"Depth frame indices mismatch in {depth_zip_path}: expected "
        f"{expected_indices[0]}..{expected_indices[-1]}, got {actual_indices}"
    )
    return np.stack([depth_frames[idx] for idx in expected_indices], axis=0)


def _load_poses_from_npz(pose_npz_path: Path, expected_frames: int) -> np.ndarray:
    """Load camera-to-world poses from an npz file."""
    data = np.load(pose_npz_path)
    inds = np.asarray(data["inds"], dtype=np.int64)
    poses = np.asarray(data["data"])

    assert inds.ndim == 1 and len(inds) == expected_frames, (
        f"Expected {expected_frames} pose indices in {pose_npz_path}, got {inds.shape}"
    )
    assert poses.shape == (expected_frames, 4, 4), (
        f"Expected poses shape {(expected_frames, 4, 4)} in {pose_npz_path}, "
        f"got {poses.shape}"
    )

    order = np.argsort(inds)
    inds = inds[order]
    poses = poses[order]
    expected_inds = np.arange(expected_frames, dtype=np.int64)
    assert np.array_equal(inds, expected_inds), (
        f"Expected pose inds 0..{expected_frames - 1} in {pose_npz_path}, "
        f"got {inds.tolist()}"
    )
    return poses.astype(np.float64)


def _load_intrinsics_from_npz(
    intrinsics_npz_path: Path,
    expected_frames: int,
) -> np.ndarray:
    """Load pinhole intrinsics from {inds, data} artifacts."""
    data = np.load(intrinsics_npz_path)
    inds = np.asarray(data["inds"], dtype=np.int64)
    intrinsics = np.asarray(data["data"])

    assert inds.ndim == 1 and len(inds) == expected_frames, (
        f"Expected {expected_frames} intrinsics indices in {intrinsics_npz_path}, "
        f"got {inds.shape}"
    )
    assert intrinsics.ndim == 2 and intrinsics.shape == (expected_frames, 4), (
        f"Expected intrinsics shape {(expected_frames, 4)} in {intrinsics_npz_path}, "
        f"got {intrinsics.shape}"
    )

    order = np.argsort(inds)
    inds = inds[order]
    intrinsics = intrinsics[order]
    expected_inds = np.arange(expected_frames, dtype=np.int64)
    assert np.array_equal(inds, expected_inds), (
        f"Expected intrinsics inds 0..{expected_frames - 1} in "
        f"{intrinsics_npz_path}, got {inds.tolist()}"
    )

    intrinsics_3x3 = np.zeros((expected_frames, 3, 3), dtype=np.float64)
    intrinsics_3x3[:, 0, 0] = intrinsics[:, 0]
    intrinsics_3x3[:, 1, 1] = intrinsics[:, 1]
    intrinsics_3x3[:, 0, 2] = intrinsics[:, 2]
    intrinsics_3x3[:, 1, 2] = intrinsics[:, 3]
    intrinsics_3x3[:, 2, 2] = 1.0
    return intrinsics_3x3


def _load_geometry_from_sample_assets(
    folder_path: Path,
    frames: np.ndarray,
) -> VideoGeometry:
    """Load geometry tensors from sample assets in the current folder."""
    num_frames = int(frames.shape[0])
    frame_h, frame_w = int(frames.shape[1]), int(frames.shape[2])
    depths = _load_depths_from_zip(folder_path / "depth.zip", num_frames)
    poses_c2w = _load_poses_from_npz(folder_path / "pose.npz", num_frames)
    intrinsics = _load_intrinsics_from_npz(folder_path / "intrinsics.npz", num_frames)

    assert depths.shape == (num_frames, frame_h, frame_w), (
        f"Depth shape mismatch in {folder_path / 'depth.zip'}: expected "
        f"{(num_frames, frame_h, frame_w)}, got {depths.shape}"
    )

    return VideoGeometry(
        frames=frames,
        depths=depths,
        intrinsics=intrinsics,
        poses_c2w=poses_c2w,
        masks=None,
        frame_indices=np.arange(num_frames, dtype=np.int32),
        original_size=(frame_h, frame_w),
        processed_size=(frame_h, frame_w),
    )


def _save_geometry_cache(
    folder_path: Path, geometry: VideoGeometry, fps: float
) -> None:
    """Persist geometry tensors to geometry.npz."""
    np.savez_compressed(
        folder_path / "geometry.npz",
        depths=geometry.depths.astype(np.float32),
        poses_c2w=geometry.poses_c2w.astype(np.float64),
        intrinsics=geometry.intrinsics.astype(np.float64),
        fps=np.array([fps], dtype=np.float32),
        original_size=np.array(geometry.original_size, dtype=np.int32),
        processed_size=np.array(geometry.processed_size, dtype=np.int32),
    )


def resize_masks(masks: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """将mask序列resize到目标尺寸"""
    H, W = target_hw
    resized = []
    for mask in masks:
        if mask.shape != (H, W):
            mask_u8 = mask.astype(np.uint8) * 255
            mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = mask_u8 > 0
        resized.append(mask)
    return np.stack(resized, axis=0)


def check_dynamic_mask_quality(
    dynamic_masks: np.ndarray, threshold: float = 0.95
) -> tuple[bool, float]:
    """检查mask质量: 如果某帧mask覆盖超过95%则认为无效(可能是全屏遮挡)"""
    if dynamic_masks is None or dynamic_masks.size == 0:
        return True, 0.0
    max_white = max(np.mean(m.astype(np.float32)) for m in dynamic_masks)
    return max_white < threshold, max_white


def log_failed_sample(log_path: Path, video_id: str, reason: str, rank: int) -> None:
    """记录处理失败的样本"""
    import datetime

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}][R{rank}] {video_id}: {reason}\n")


def cleanup_failed_output(output_dir: Path) -> None:
    """清理失败的输出目录"""
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)


def mark_as_failed(output_dir: Path, reason: str) -> None:
    """创建.skip标记文件，后续运行时跳过此文件夹"""
    import datetime

    output_dir.mkdir(parents=True, exist_ok=True)
    skip_file = output_dir / ".skip"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    skip_file.write_text(f"[{ts}] {reason}\n", encoding="utf-8")


def is_marked_failed(output_dir: Path) -> tuple[bool, str]:
    """检查文件夹是否已被标记为失败"""
    skip_file = output_dir / ".skip"
    if skip_file.exists():
        try:
            reason = skip_file.read_text(encoding="utf-8").strip()
            return True, reason
        except Exception:
            return True, "unknown"
    return False, ""


# ============================================================================
# 日志函数
# ============================================================================


def log_stage_header(stage_name: str, rank: int) -> None:
    """打印阶段开始的header"""
    import datetime

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n{'=' * 60}", flush=True)
    print(f"[{ts}][R{rank}] STAGE: {stage_name}", flush=True)
    print(f"{'=' * 60}", flush=True)


def log_stage_progress(
    stage: str,
    idx: int,
    total: int,
    video_id: str,
    rank: int,
    status: str = "",
    skipped: int = 0,
) -> None:
    import datetime

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    skip_info = f" (skipped={skipped})" if skipped > 0 else ""
    status_str = f" [{status}]" if status else ""
    print(
        f"[{ts}][R{rank}][{stage}] {idx}/{total}{skip_info} {video_id}{status_str}",
        flush=True,
    )


def log_stage_summary(
    stage: str, rank: int, processed: int, skipped: int, failed: int = 0
) -> None:
    import datetime

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    parts = [f"processed={processed}", f"skipped={skipped}"]
    if failed > 0:
        parts.append(f"failed={failed}")
    print(f"[{ts}][R{rank}][{stage}] DONE: {', '.join(parts)}", flush=True)


def log_message(
    msg: str, rank: int, stage: str = None, only_rank0: bool = False
) -> None:
    if only_rank0 and rank != 0:
        return
    import datetime

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    tag = f"[{ts}][R{rank}]"
    if stage:
        tag += f"[{stage}]"
    print(f"{tag} {msg}", flush=True)


# ============================================================================
# Clip discovery and completeness
# ============================================================================


def _is_sample_folder_name(name: str) -> bool:
    """Return whether the folder name matches the standard sample id format."""
    return len(name) == 8 and name.isdigit()


def discover_available_clips(
    cfg,
    output_root: Path,
    rank: int,
) -> list[tuple[Path, str, Path]]:
    """Discover valid clip folders under output_root in stable order."""
    if not output_root.exists():
        return []

    clip_items: list[tuple[Path, str, Path]] = []
    invalid_count = 0
    ignored_count = 0

    for folder_path in sorted(output_root.iterdir(), key=lambda path: path.name):
        if not folder_path.is_dir():
            continue
        folder_id = folder_path.name
        if not _is_sample_folder_name(folder_id):
            ignored_count += 1
            continue
        if not check_clip_complete(folder_path, cfg):
            invalid_count += 1
            continue
        clip_items.append((folder_path / CLIP_FILENAME, folder_id, folder_path))

    log_message(
        f"Discovered {len(clip_items)} valid clips, ignored={ignored_count}, invalid={invalid_count}",
        rank,
        "Init",
        only_rank0=True,
    )
    return clip_items


def _check_stage3_geometry_complete(video_out: Path) -> bool:
    """检查geometry.npz是否已生成"""
    return (video_out / "geometry.npz").exists()


def _check_folder_complete(folder_path: Path, cfg) -> tuple[bool, str]:
    """检查文件夹是否完整 (clip + geometry + samples都存在)"""
    is_failed, fail_reason = is_marked_failed(folder_path)
    if is_failed:
        return False, f"error:{fail_reason}"

    if not check_clip_complete(folder_path, cfg):
        return False, "interrupted:clip_incomplete"

    if not _check_stage3_geometry_complete(folder_path):
        return False, "interrupted:geometry_incomplete"

    if not _check_stage3_samples_complete(folder_path, cfg):
        return False, "interrupted:samples_incomplete"

    return True, "ok"


def _check_stage3_samples_complete(video_out: Path, cfg) -> bool:
    """检查训练样本是否完整生成"""
    names = get_sample_naming(cfg.naming_style)
    prefixes = (
        ["train_"]
        if cfg.num_samples == 1
        else [f"train_sample{i:03d}_" for i in range(cfg.num_samples)]
    )
    use_latent_point_cloud = cfg.point_cloud_type == "latent"
    has_rgb_proj = (not use_latent_point_cloud) and ("rgb" in cfg.projection_channels)

    for prefix in prefixes:
        required = [
            video_out / f"{prefix}sample.json",
            video_out / f"{prefix}{names.preceding_rgb}.mp4",
            video_out / f"{prefix}{names.target_rgb}.mp4",
        ]
        if use_latent_point_cloud:
            required.extend(
                [
                    video_out / f"{prefix}{names.preceding_scene_proj_rgb}.pt",
                    video_out / f"{prefix}{names.target_scene_proj_rgb}.pt",
                ]
            )
        if has_rgb_proj:
            required.extend(
                [
                    video_out / f"{prefix}{names.preceding_scene_proj_rgb}.mp4",
                    video_out / f"{prefix}{names.target_scene_proj_rgb}.mp4",
                ]
            )
        if not all(p.exists() for p in required):
            return False
    return True


# ============================================================================
# 主处理流程
# ============================================================================


def run_pipeline_iteration(
    cfg,
    rank: int,
    local_rank: int,
    device: str,
    output_root: Path,
    video_items: list[tuple[Path, str, Path]],
    iteration: int,
) -> int:
    """
    Run one downstream iteration and return the remaining interrupted count.

    Processing stages:
    1. Load foreground masks from sample assets
    2. Load geometry from sample assets
    3. Build training samples
    """
    if not video_items:
        log_message("No video items to process!", rank, "Init")
        return 0

    _ = local_rank

    log_stage_header("Mask Import", rank)
    mask_failed_log = output_root / "mask_failed_samples.txt"
    mask_processed, mask_skipped, mask_failed = 0, 0, 0

    for idx, (video_path, folder_id, folder_path) in enumerate(video_items, start=1):
        is_failed, _ = is_marked_failed(folder_path)
        if is_failed:
            mask_skipped += 1
            continue

        masks_npy = folder_path / "dynamic_masks.npy"
        masks_mp4 = folder_path / "dynamic_masks.mp4"

        if cfg.skip_existing and masks_npy.exists():
            try:
                dynamic_masks = np.asarray(np.load(masks_npy)).astype(bool)
                assert dynamic_masks.ndim == 3, (
                    f"dynamic_masks.npy must be 3D, got {dynamic_masks.shape}"
                )
                if not masks_mp4.exists():
                    masks_rgb = np.repeat(
                        dynamic_masks.astype(np.uint8)[..., None] * 255, 3, axis=-1
                    )
                    save_video(masks_rgb, masks_mp4, fps=get_video_fps(video_path))
                mask_skipped += 1
                continue
            except Exception:
                pass

        try:
            hw = _get_video_hw(video_path)
            assert hw is not None, f"Cannot read video size from {video_path}"
            num_frames = get_video_frame_count(video_path)
            dynamic_masks = _load_masks_from_zip(
                folder_path / "mask.zip",
                expected_frames=num_frames,
                expected_hw=hw,
            )
            np.save(masks_npy, dynamic_masks)
            masks_rgb = np.repeat(
                dynamic_masks.astype(np.uint8)[..., None] * 255,
                3,
                axis=-1,
            )
            save_video(masks_rgb, masks_mp4, fps=get_video_fps(video_path))
            mask_processed += 1
            log_stage_progress(
                "Masks",
                idx,
                len(video_items),
                folder_id,
                rank,
                status="loaded",
                skipped=mask_skipped,
            )
        except Exception as e:
            log_stage_progress(
                "Masks",
                idx,
                len(video_items),
                folder_id,
                rank,
                status=f"ERROR: {str(e)[:50]}",
                skipped=mask_skipped,
            )
            log_failed_sample(mask_failed_log, folder_id, str(e), rank)
            mark_as_failed(folder_path, f"Mask import error: {e}")
            mask_failed += 1

        if cfg.cleanup_interval and idx % cfg.cleanup_interval == 0:
            _cleanup_memory()

    log_stage_summary("Masks", rank, mask_processed, mask_skipped, mask_failed)
    _cleanup_memory()

    log_stage_header("Geometry + Sample Building", rank)

    sample_config = SampleConfig()
    point_cloud_vae = None
    latent_projection_stride = 1
    if sample_config.point_cloud_type == "latent":
        from latent_mem.wrapper.wan.base import WanVAEWrapper

        vae_dtype = _get_point_cloud_vae_dtype(sample_config.point_cloud_vae_dtype)
        point_cloud_vae = WanVAEWrapper(
            wan_model_path=sample_config.point_cloud_vae_model_path,
            vae_checkpoint=sample_config.point_cloud_vae_checkpoint,
            device=device,
            dtype=vae_dtype,
        ).to(device=device, dtype=vae_dtype)
        latent_projection_stride = int(point_cloud_vae.vae_stride[0])

    failed_log = output_root / "failed_samples.txt"
    s3_processed, s3_skipped, s3_failed = 0, 0, 0

    for idx, (video_path, folder_id, folder_path) in enumerate(video_items, start=1):
        is_failed, fail_reason = is_marked_failed(folder_path)
        if is_failed:
            s3_skipped += 1
            continue

        geo_complete = (
            _check_stage3_geometry_complete(folder_path) if cfg.skip_existing else False
        )
        sample_complete = (
            _check_stage3_samples_complete(folder_path, cfg)
            if cfg.skip_existing
            else False
        )

        if geo_complete and sample_complete:
            s3_skipped += 1
            continue

        try:
            folder_path.mkdir(parents=True, exist_ok=True)
            fps = cfg.fps_override or get_video_fps(video_path)
            frames = np.asarray(load_video_frames(video_path))

            masks_npy = folder_path / "dynamic_masks.npy"
            if masks_npy.exists():
                dynamic_masks = np.asarray(np.load(masks_npy)).astype(bool)
            else:
                hw = _get_video_hw(video_path)
                assert hw is not None, f"Cannot read video size from {video_path}"
                dynamic_masks = _load_masks_from_zip(
                    folder_path / "mask.zip",
                    expected_frames=int(frames.shape[0]),
                    expected_hw=hw,
                )

            valid, max_white = check_dynamic_mask_quality(dynamic_masks, 0.95)
            if not valid:
                log_stage_progress(
                    "Stage3",
                    idx,
                    len(video_items),
                    folder_id,
                    rank,
                    status=f"SKIP: mask {max_white:.0%} white",
                    skipped=s3_skipped,
                )
                log_failed_sample(
                    failed_log, folder_id, f"mask white ratio {max_white:.0%}", rank
                )
                mark_as_failed(folder_path, f"mask white ratio {max_white:.0%}")
                s3_failed += 1
                continue

            if not geo_complete:
                geometry = _load_geometry_from_sample_assets(folder_path, frames)
                _save_geometry_cache(folder_path, geometry, fps)
                H, W = geometry.frames.shape[1:3]
                dynamic_masks_proc = (
                    resize_masks(dynamic_masks, (H, W))
                    if dynamic_masks.size > 0
                    else None
                )
                geo_status = "geo"
            else:
                with np.load(folder_path / "geometry.npz") as geo_data:
                    H, W = (
                        int(geo_data["processed_size"][0]),
                        int(geo_data["processed_size"][1]),
                    )
                    depths = np.asarray(geo_data["depths"])
                    intrinsics = np.asarray(geo_data["intrinsics"])
                    poses_c2w = np.asarray(geo_data["poses_c2w"])
                    original_size = tuple(int(x) for x in geo_data["original_size"])
                frames_resized = np.stack(
                    [
                        cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
                        for f in frames
                    ],
                    axis=0,
                )
                geometry = VideoGeometry(
                    frames=frames_resized,
                    depths=depths,
                    intrinsics=intrinsics,
                    poses_c2w=poses_c2w,
                    masks=None,
                    frame_indices=np.arange(len(frames), dtype=np.int32),
                    original_size=original_size,
                    processed_size=(H, W),
                )
                dynamic_masks_proc = (
                    resize_masks(dynamic_masks, (H, W))
                    if dynamic_masks.size > 0
                    else None
                )
                geo_status = "geo-cached"

            # 构建训练样本
            ref_status = ""
            if not sample_complete:
                original_frames = frames
                prefixes = (
                    ["train_"]
                    if cfg.num_samples == 1
                    else [f"train_sample{i:03d}_" for i in range(cfg.num_samples)]
                )

                # Reuse the fixed indices produced by run_video_collect.
                for prefix in prefixes:
                    sample_json = folder_path / f"{prefix}sample.json"
                    assert sample_json.exists(), (
                        f"Missing required sample metadata: {sample_json}"
                    )
                    sample_meta = json.loads(sample_json.read_text(encoding="utf-8"))

                    sample_indices = SampleIndices(
                        t0=int(sample_meta["t0"]),
                        preceding_indices=[int(idx) for idx in sample_meta["P_idx"]],
                        target_indices=[int(idx) for idx in sample_meta["T_idx"]],
                        candidate_indices=[int(idx) for idx in sample_meta["C_idx"]],
                    )
                    scene_idx = int(sample_meta["scene_idx"])
                    assert scene_idx == int(sample_indices.t0), (
                        f"scene_idx mismatch in {sample_json}: "
                        f"{scene_idx} vs t0={sample_indices.t0}"
                    )
                    output_size = tuple(int(x) for x in sample_meta["output_size"])
                    assert len(output_size) == 2, (
                        f"output_size must be [H, W] in {sample_json}"
                    )

                    scene_latent = None
                    if sample_config.point_cloud_type == "latent":
                        assert point_cloud_vae is not None
                        scene_frame = np.asarray(frames[scene_idx])
                        scene_latent = _encode_scene_latent(
                            frame=scene_frame,
                            vae=point_cloud_vae,
                            device=device,
                        )

                    # 调用build_training_sample构建完整训练样本
                    sample = build_training_sample(
                        geometry=geometry,
                        config=sample_config,
                        dynamic_masks=dynamic_masks_proc,
                        projection_fill_kernel=cfg.projection_fill_kernel,
                        original_frames=original_frames,
                        output_size=output_size,
                        indices=sample_indices,
                        scene_latent=scene_latent,
                        latent_projection_stride=latent_projection_stride,
                    )
                    source_video_path = load_source_video_path(folder_path)
                    if source_video_path is not None:
                        sample["meta"]["source_video_path"] = source_video_path

                    meta = sample.get("meta", {})
                    r_stats = meta.get("R_stats", {})
                    ref_count = len(meta.get("R_idx", []))
                    best_iou = r_stats.get("best_iou", 0)
                    threshold = r_stats.get("threshold", 0)

                    if ref_count > 0:
                        ref_status = f"ref={ref_count}, iou={best_iou:.3f}"
                    else:
                        reason = r_stats.get("no_ref_reason", "unknown")
                        if reason == "iou_below_threshold":
                            ref_status = f"NO_REF(iou={best_iou:.3f}<{threshold})"
                        elif reason == "no_candidates":
                            ref_status = "NO_REF(no_candidates)"
                        else:
                            ref_status = f"NO_REF({reason})"

                    save_training_sample(
                        sample,
                        folder_path,
                        projection_channels=sample["meta"]["projection_channels"],
                        fps=fps,
                        naming=cfg.naming_style,
                        name_prefix=prefix,
                    )
                sample_status = f"sample({ref_status})"
            else:
                sample_json = folder_path / "train_sample.json"
                if sample_json.exists():
                    try:
                        sample_meta = json.loads(
                            sample_json.read_text(encoding="utf-8")
                        )
                        ref_count = len(sample_meta.get("R_idx", []))
                        r_stats = sample_meta.get("R_stats", {})
                        if ref_count > 0:
                            ref_status = (
                                f"ref={ref_count}, iou={r_stats.get('best_iou', 0):.3f}"
                            )
                        else:
                            reason = r_stats.get("no_ref_reason", "unknown")
                            best_iou = r_stats.get("best_iou", 0)
                            threshold = r_stats.get("threshold", 0)
                            if reason == "iou_below_threshold":
                                ref_status = f"NO_REF(iou={best_iou:.3f}<{threshold})"
                            else:
                                ref_status = f"NO_REF({reason})"
                    except Exception:
                        ref_status = "ref=?"
                sample_status = f"sample-cached({ref_status})"

            s3_processed += 1
            log_stage_progress(
                "Stage3",
                idx,
                len(video_items),
                folder_id,
                rank,
                status=f"{geo_status}+{sample_status}",
                skipped=s3_skipped,
            )

        except Exception as e:
            tb = traceback.format_exc()
            log_stage_progress(
                "Stage3",
                idx,
                len(video_items),
                folder_id,
                rank,
                status=f"ERROR: {str(e)[:50]}",
                skipped=s3_skipped,
            )
            log_failed_sample(failed_log, folder_id, f"{e}\n{tb}", rank)
            mark_as_failed(folder_path, f"Stage3 error: {e}")
            if not geo_complete:
                skip_file = folder_path / ".skip"
                skip_content = (
                    skip_file.read_text(encoding="utf-8")
                    if skip_file.exists()
                    else None
                )
                cleanup_failed_output(folder_path)
                if skip_content:
                    folder_path.mkdir(parents=True, exist_ok=True)
                    skip_file.write_text(skip_content, encoding="utf-8")
            s3_failed += 1
            continue

        if cfg.cleanup_interval and idx % cfg.cleanup_interval == 0:
            _cleanup_memory()

    log_stage_summary("Stage3", rank, s3_processed, s3_skipped, s3_failed)
    if point_cloud_vae is not None:
        del point_cloud_vae
    _cleanup_memory()

    log_stage_header(f"Completeness Check (iter={iteration})", rank)

    error_folders = []
    interrupted_folders = []
    complete_count = 0

    for _, folder_id, folder_path in video_items:
        is_complete, reason = _check_folder_complete(folder_path, cfg)
        if is_complete:
            complete_count += 1
        elif reason.startswith("error:"):
            error_folders.append((folder_id, reason))
        else:
            interrupted_folders.append((folder_id, reason))

    log_message(
        f"Completeness: complete={complete_count}, errors={len(error_folders)}, interrupted={len(interrupted_folders)}",
        rank,
        "Check",
    )

    if error_folders and (not cfg.quiet_nonzero or rank == 0):
        log_message(
            f"Keeping {len(error_folders)} failed folders with .skip markers for manual review...",
            rank,
            "Check",
        )
        for folder_id, reason in error_folders:
            log_message(f"  {folder_id}: {reason}", rank, "Check")

    if interrupted_folders and (not cfg.quiet_nonzero or rank == 0):
        log_message(
            f"Keeping {len(interrupted_folders)} interrupted folders for incremental processing...",
            rank,
            "Check",
        )
        for folder_id, reason in interrupted_folders:
            log_message(f"  {folder_id}: {reason}", rank, "Check")

    return len(interrupted_folders)


def main():
    cfg = CONFIG

    rank, world_size, local_rank = get_rank_info()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if rank == 0:
        cfg.print_config()

    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    log_stage_header("Clip Discovery", rank)
    discovered_items = discover_available_clips(cfg, output_root, rank)
    if not discovered_items:
        log_message("No valid clips found under output_root!", rank, "Init")
        return

    my_video_items = shard_items(discovered_items, rank, world_size)
    log_message(f"This rank processing {len(my_video_items)} clips", rank, "Init")

    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        log_stage_header(f"PIPELINE ITERATION {iteration}", rank)

        incomplete_count = run_pipeline_iteration(
            cfg=cfg,
            rank=rank,
            local_rank=local_rank,
            device=device,
            output_root=output_root,
            video_items=my_video_items,
            iteration=iteration,
        )

        if incomplete_count == 0:
            log_message(
                f"All {len(my_video_items)} assigned clips complete!", rank, "Done"
            )
            break

        log_message(
            f"Iteration {iteration} done. {incomplete_count} clip folders remain interrupted, retrying...",
            rank,
            "Retry",
        )
        iteration += 1

    if iteration >= max_iterations:
        log_message(
            f"WARNING: Reached max iterations ({max_iterations}), some folders may be incomplete",
            rank,
            "Done",
        )

    import datetime

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n{'=' * 60}", flush=True)
    print(
        f"[{ts}][R{rank}] ALL STAGES COMPLETE (iterations={iteration + 1})", flush=True
    )
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
