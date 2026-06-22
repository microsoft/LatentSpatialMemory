#!/usr/bin/env python3
"""
Resize existing training videos to target resolution (480x832).

This script fixes videos that were saved at MapAnything's processed resolution
instead of the target clip resolution.

Usage:
    python -m data_process.resize_existing_videos
    python -m data_process.resize_existing_videos --data-root data/Spatia/frame33_fps16_2000
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

from data_process.video_io import save_mp4

# Videos to resize (these are the training outputs that may be at wrong resolution)
VIDEOS_TO_RESIZE = [
    "train_target_rgb.mp4",
    "train_target_scene_proj_rgb.mp4",
    "train_preceding_rgb.mp4",
    "train_preceding_scene_proj_rgb.mp4",
    "train_reference_rgb.mp4",
]

# Target resolution
TARGET_WIDTH = 832
TARGET_HEIGHT = 480


def get_video_resolution(video_path: Path) -> tuple[int, int] | None:
    """Get video resolution (width, height). Returns None if can't read."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (width, height)
    except Exception:
        return None


def resize_video(
    video_path: Path, target_w: int, target_h: int, fps: float = 16.0
) -> bool:
    """
    Resize video to target resolution in-place.
    Returns True if resized, False if skipped or error.
    """
    try:
        # Read video
        frames = iio.imread(str(video_path), plugin="pyav")
        if frames.ndim != 4 or frames.shape[-1] != 3:
            return False

        current_h, current_w = frames.shape[1], frames.shape[2]

        # Skip if already at target resolution
        if current_w == target_w and current_h == target_h:
            return False

        # Resize frames
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(
                frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
            resized_frames.append(resized)
        resized_frames = np.stack(resized_frames, axis=0)

        # Save back (backup original first)
        backup_path = video_path.with_suffix(".mp4.bak")
        shutil.move(str(video_path), str(backup_path))

        try:
            save_mp4(resized_frames, video_path, fps=fps)
            # Remove backup on success
            backup_path.unlink()
            return True
        except Exception:
            # Restore backup on failure
            shutil.move(str(backup_path), str(video_path))
            raise

    except Exception as e:
        print(f"  Error resizing {video_path}: {e}")
        return False


def list_sample_folders(data_root: Path) -> list[Path]:
    """List all sample folders (8-digit numbered folders)."""
    folders = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name.isdigit() and len(p.name) == 8:
            folders.append(p)
    return folders


def main():
    parser = argparse.ArgumentParser(
        description="Resize existing training videos to target resolution"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/Spatia/frame33_fps16_2000",
        help="Root directory containing sample folders",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=TARGET_WIDTH,
        help="Target width",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=TARGET_HEIGHT,
        help="Target height",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=16.0,
        help="Video FPS",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be resized, don't actually resize",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    target_w = args.target_width
    target_h = args.target_height

    print("=" * 60)
    print("Video Resizer")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Target resolution: {target_w}x{target_h}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 60)

    folders = list_sample_folders(data_root)
    print(f"Found {len(folders)} sample folders")

    total_resized = 0
    total_skipped = 0
    total_missing = 0

    for folder in tqdm(folders, desc="Processing folders"):
        for video_name in VIDEOS_TO_RESIZE:
            video_path = folder / video_name
            if not video_path.exists():
                total_missing += 1
                continue

            resolution = get_video_resolution(video_path)
            if resolution is None:
                continue

            current_w, current_h = resolution
            if current_w == target_w and current_h == target_h:
                total_skipped += 1
                continue

            if args.dry_run:
                print(
                    f"Would resize: {video_path} ({current_w}x{current_h} -> {target_w}x{target_h})"
                )
                total_resized += 1
            else:
                if resize_video(video_path, target_w, target_h, args.fps):
                    total_resized += 1
                else:
                    total_skipped += 1

    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Resized: {total_resized}")
    print(f"  Skipped (already correct): {total_skipped}")
    print(f"  Missing: {total_missing}")
    print("=" * 60)


if __name__ == "__main__":
    main()
