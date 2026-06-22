#!/usr/bin/env python3
"""
Spatia clip sampling entrypoint.

This script only prepares clip folders under output_root:
1. Collect source videos
2. Sample fixed-length clips into folder_id/clip.mp4
3. Persist target-only training outputs for downstream captioning
4. Persist source_video_path.txt for downstream stages
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_process._0_0_0_root_assign import CONFIG, SampleConfig
from data_process.clip_sampling import (
    CLIP_FILENAME,
    check_clip_complete,
    check_ffmpeg_readiness,
    extract_clip,
    load_source_video_path,
    normalize_source_video_path,
    prepare_folder_ids,
    save_source_video_path,
    shuffle_videos,
)
from data_process.dataset_writer import save_video
from data_process.distributed import get_rank_info, shard_items
from data_process.naming import get_sample_naming
from data_process.sample_builder import resize_frames
from data_process.sample_indices import sample_frame_indices
from data_process.video_io import (
    get_video_fps,
    list_video_files,
    load_video_frames,
)


def log_stage_header(stage_name: str, rank: int) -> None:
    import datetime

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n{'=' * 60}", flush=True)
    print(f"[{ts}][R{rank}] STAGE: {stage_name}", flush=True)
    print(f"{'=' * 60}", flush=True)


def log_stage_progress(
    stage: str,
    idx: int,
    total: int,
    folder_id: str,
    rank: int,
    status: str = "",
    skipped: int = 0,
) -> None:
    import datetime

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    skip_info = f" (skipped={skipped})" if skipped > 0 else ""
    status_str = f" [{status}]" if status else ""
    print(
        f"[{ts}][R{rank}][{stage}] {idx}/{total}{skip_info} {folder_id}{status_str}",
        flush=True,
    )


def log_message(
    msg: str, rank: int, stage: str | None = None, only_rank0: bool = False
) -> None:
    if only_rank0 and rank != 0:
        return
    import datetime

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    tag = f"[{ts}][R{rank}]"
    if stage:
        tag += f"[{stage}]"
    print(f"{tag} {msg}", flush=True)


def is_marked_failed(output_dir: Path) -> tuple[bool, str]:
    """Check whether the folder has a failure marker."""
    skip_file = output_dir / ".skip"
    if skip_file.exists():
        try:
            reason = skip_file.read_text(encoding="utf-8").strip()
            return True, reason
        except Exception:
            return True, "unknown"
    return False, ""


def log_failed_folder_skip(folder_path: Path, rank: int, reason: str) -> None:
    """Warn and skip folders that were marked as failed by earlier stages."""
    reason_suffix = f": {reason}" if reason else ""
    log_message(
        f"WARNING: skipping failed folder {folder_path.name}{reason_suffix}",
        rank,
        "Cleanup",
    )


def collect_all_source_videos(video_dirs: list[str], rank: int) -> list[Path]:
    """Collect all source videos from configured directories."""
    all_videos: list[Path] = []
    for video_dir in video_dirs:
        root = Path(video_dir)
        if not root.exists():
            log_message(f"Warning: video_dir not found: {video_dir}", rank, "Init")
            continue
        videos = list(list_video_files(root))
        log_message(
            f"Found {len(videos)} videos in {video_dir}",
            rank,
            "Init",
            only_rank0=True,
        )
        all_videos.extend(videos)
    return sorted(all_videos)


def is_sample_folder_name(name: str) -> bool:
    """Return whether the folder name matches the standard sample id format."""
    return len(name) == 8 and name.isdigit()


def discover_assigned_source_videos(
    output_root: Path,
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Load canonical source-video assignments from existing sample folders."""
    assigned_by_folder: dict[str, Path] = {}
    duplicate_folders_by_source: dict[str, list[str]] = {}

    if not output_root.exists():
        return assigned_by_folder, duplicate_folders_by_source

    source_owner_by_key: dict[str, str] = {}
    for folder_path in sorted(output_root.iterdir(), key=lambda path: path.name):
        if not folder_path.is_dir():
            continue
        folder_id = folder_path.name
        if not is_sample_folder_name(folder_id):
            continue

        source_video_path = load_source_video_path(folder_path)
        if not source_video_path:
            continue

        source_key = normalize_source_video_path(source_video_path)
        owner_folder_id = source_owner_by_key.get(source_key)
        if owner_folder_id is None:
            source_owner_by_key[source_key] = folder_id
            assigned_by_folder[folder_id] = Path(source_key)
            continue

        duplicate_folders = duplicate_folders_by_source.setdefault(
            source_key,
            [owner_folder_id],
        )
        duplicate_folders.append(folder_id)

    return assigned_by_folder, duplicate_folders_by_source


def _target_sample_prefixes(num_samples: int) -> list[str]:
    """Return the training-sample file prefixes for one folder."""
    return (
        ["train_"]
        if num_samples == 1
        else [f"train_sample{i:03d}_" for i in range(num_samples)]
    )


def _check_target_samples_complete(folder_path: Path, cfg) -> bool:
    """Check whether target-only outputs already exist for all samples."""
    if not check_clip_complete(folder_path, cfg):
        return False

    names = get_sample_naming(cfg.naming_style)
    for prefix in _target_sample_prefixes(cfg.num_samples):
        sample_json = folder_path / f"{prefix}sample.json"
        target_video = folder_path / f"{prefix}{names.target_rgb}.mp4"
        if not sample_json.exists() or not target_video.exists():
            return False
    return True


def run_clip_sampling_iteration(
    cfg,
    rank: int,
    world_size: int,
    output_root: Path,
    all_videos: list[Path],
    my_folder_ids: list[str],
    iteration: int,
) -> int:
    """Sample clips for one iteration and return the remaining incomplete count."""
    failed_folder_ids: set[str] = set()
    for folder_id in my_folder_ids:
        folder_path = output_root / folder_id
        is_failed, reason = is_marked_failed(folder_path)
        if is_failed:
            failed_folder_ids.add(folder_id)
            log_failed_folder_skip(folder_path, rank, reason)

    assigned_source_by_folder, duplicate_folders_by_source = (
        discover_assigned_source_videos(output_root)
    )
    if duplicate_folders_by_source:
        for source_key, folder_ids in duplicate_folders_by_source.items():
            log_message(
                f"Duplicate source assignment detected for {source_key}: {folder_ids}",
                rank,
                "Init",
                only_rank0=True,
            )

    assigned_source_keys = {
        source_path.as_posix() for source_path in assigned_source_by_folder.values()
    }
    pool_seed = (cfg.shuffle_seed or 42) + (iteration * 1000)
    shuffled_videos = shuffle_videos(all_videos, pool_seed)
    available_videos = [
        video
        for video in shuffled_videos
        if normalize_source_video_path(video) not in assigned_source_keys
    ]
    my_available_videos = shard_items(available_videos, rank, world_size)
    clip_target_wh = (cfg.clip_target_width, cfg.clip_target_height)
    clip_seed = (cfg.shuffle_seed or 42) + iteration

    log_stage_header(f"Clip Extraction (iter={iteration})", rank)
    log_message(
        f"Assigned videos={len(assigned_source_by_folder)}, "
        f"available unique videos={len(available_videos)}, "
        f"this rank candidates={len(my_available_videos)}",
        rank,
        "Init",
        only_rank0=True,
    )

    setup_extracted = 0
    setup_cached = 0
    setup_skipped = 0
    video_pool_idx = 0

    for folder_idx, folder_id in enumerate(my_folder_ids, start=1):
        folder_path = output_root / folder_id
        clip_path = folder_path / CLIP_FILENAME

        if folder_id in failed_folder_ids:
            setup_skipped += 1
            log_stage_progress(
                "Extract",
                folder_idx,
                len(my_folder_ids),
                folder_id,
                rank,
                status="WARNING: marked failed, skipped",
                skipped=setup_cached + setup_skipped,
            )
            continue

        clip_complete = check_clip_complete(folder_path, cfg)
        target_complete = _check_target_samples_complete(folder_path, cfg)
        if clip_complete and target_complete:
            setup_cached += 1
            continue

        if not clip_complete and clip_path.exists():
            clip_path.unlink(missing_ok=True)

        extraction_success = False
        attempts = 0
        max_attempts = len(my_available_videos) + 1
        last_failure_message = ""
        assigned_source_video = assigned_source_by_folder.get(folder_id)
        assigned_source_consumed = False

        while not extraction_success and attempts < max_attempts:
            if not assigned_source_consumed and assigned_source_video is not None:
                source_video = assigned_source_video
                assigned_source_consumed = True
            else:
                if video_pool_idx >= len(my_available_videos):
                    break
                source_video = my_available_videos[video_pool_idx]
                video_pool_idx += 1
            attempts += 1

            success, message = extract_clip(
                video_path=source_video,
                output_path=clip_path,
                num_frames=cfg.clip_num_frames,
                target_fps=cfg.clip_target_fps,
                target_wh=clip_target_wh,
                seed=clip_seed,
            )
            if not success:
                last_failure_message = message
                if message.startswith("ffmpeg"):
                    log_message(
                        f"{folder_id} failed on {source_video.name}: {message}",
                        rank,
                        "Extract",
                    )
                elif attempts <= 3:
                    log_message(
                        f"{folder_id} skipped {source_video.name}: {message}",
                        rank,
                        "Extract",
                    )
                continue

            save_source_video_path(folder_path, source_video)
            assigned_source_by_folder[folder_id] = source_video.resolve()
            setup_extracted += 1
            extraction_success = True
            clip_complete = True
            break

        if not clip_complete:
            setup_skipped += 1
            reason_suffix = (
                f"; last_error={last_failure_message}" if last_failure_message else ""
            )
            log_message(
                f"WARN: {folder_id} - exhausted video pool after {attempts} attempts"
                f"{reason_suffix}",
                rank,
                "Extract",
            )
            continue

        try:
            clip_path = folder_path / CLIP_FILENAME
            frames = np.asarray(load_video_frames(clip_path))
            fps = cfg.fps_override or get_video_fps(clip_path)
            sample_config = SampleConfig()
            names = get_sample_naming(cfg.naming_style)
            source_video_path = load_source_video_path(folder_path)
            rng = np.random.default_rng(cfg.sample_random_seed)

            # Fix train_sample sampling at collect time so later stages only consume it.
            for prefix in _target_sample_prefixes(cfg.num_samples):
                sample_indices = sample_frame_indices(
                    num_frames=int(frames.shape[0]),
                    N_target=sample_config.N_target,
                    M_pre=sample_config.M_pre,
                    min_gap_for_candidates=sample_config.min_gap_for_candidates,
                    rng=rng,
                )
                target_rgb = resize_frames(
                    frames[sample_indices.target_indices],
                    (cfg.clip_target_height, cfg.clip_target_width),
                )
                meta = {
                    "t0": int(sample_indices.t0),
                    "P_idx": list(sample_indices.preceding_indices),
                    "T_idx": list(sample_indices.target_indices),
                    "C_idx": list(sample_indices.candidate_indices),
                    "scene_idx": int(sample_indices.t0),
                    "output_size": (
                        cfg.clip_target_height,
                        cfg.clip_target_width,
                    ),
                    "naming": cfg.naming_style,
                }
                if source_video_path is not None:
                    meta["source_video_path"] = source_video_path

                save_video(
                    target_rgb,
                    folder_path / f"{prefix}{names.target_rgb}.mp4",
                    fps=fps,
                )
                (folder_path / f"{prefix}sample.json").write_text(
                    json.dumps(meta, indent=2),
                    encoding="utf-8",
                )

            status = "ok"
            if attempts > 1:
                status = f"ok (attempts={attempts})"
            elif attempts == 0 and not target_complete:
                status = "target-only"
            log_stage_progress(
                "Extract",
                folder_idx,
                len(my_folder_ids),
                folder_id,
                rank,
                status=status,
                skipped=setup_cached + setup_skipped,
            )
        except Exception as exc:
            setup_skipped += 1
            log_message(
                f"Failed to build target-only outputs for {folder_id}: {exc}",
                rank,
                "Target",
            )
            log_stage_progress(
                "Extract",
                folder_idx,
                len(my_folder_ids),
                folder_id,
                rank,
                status=f"ERROR: {str(exc)[:50]}",
                skipped=setup_cached + setup_skipped,
            )

    log_message(
        f"Clip extraction: extracted={setup_extracted}, cached={setup_cached}, "
        f"skipped={setup_skipped}, videos_used={video_pool_idx}/{len(my_available_videos)}",
        rank,
        "Extract",
    )

    incomplete_count = 0
    for folder_id in my_folder_ids:
        if folder_id in failed_folder_ids:
            continue
        if not _check_target_samples_complete(output_root / folder_id, cfg):
            incomplete_count += 1
    return incomplete_count


def main() -> None:
    cfg = CONFIG
    rank, world_size, _ = get_rank_info()

    if rank == 0:
        cfg.print_config()

    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    log_stage_header("Video Collection", rank)
    ffmpeg_ready, ffmpeg_message = check_ffmpeg_readiness()
    if not ffmpeg_ready:
        log_message(f"FFmpeg unavailable: {ffmpeg_message}", rank, "Init")
        return
    log_message(
        f"Using ffmpeg executable: {ffmpeg_message}",
        rank,
        "Init",
        only_rank0=True,
    )

    all_videos = collect_all_source_videos(cfg.video_dirs, rank)
    log_message(
        f"Total source videos: {len(all_videos)}", rank, "Init", only_rank0=True
    )
    if not all_videos:
        log_message("No source videos found!", rank, "Init")
        return

    max_videos = cfg.max_videos or len(all_videos)
    folder_ids = prepare_folder_ids(max_videos)
    my_folder_ids = shard_items(folder_ids, rank, world_size)
    log_message(f"This rank processing {len(my_folder_ids)} folders", rank, "Init")

    max_iterations = 10
    iteration = 0
    while iteration < max_iterations:
        log_stage_header(f"CLIP SAMPLING ITERATION {iteration}", rank)
        incomplete_count = run_clip_sampling_iteration(
            cfg=cfg,
            rank=rank,
            world_size=world_size,
            output_root=output_root,
            all_videos=all_videos,
            my_folder_ids=my_folder_ids,
            iteration=iteration,
        )
        if incomplete_count == 0:
            log_message(f"All {len(my_folder_ids)} folders have clips!", rank, "Done")
            break
        log_message(
            f"Iteration {iteration} done. {incomplete_count} folders still missing clips, retrying...",
            rank,
            "Retry",
        )
        iteration += 1

    if iteration >= max_iterations:
        log_message(
            f"WARNING: Reached max iterations ({max_iterations}), some clips may be missing",
            rank,
            "Done",
        )

    import datetime

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n{'=' * 60}", flush=True)
    print(f"[{ts}][R{rank}] CLIP SAMPLING COMPLETE (iterations={iteration + 1})")
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
