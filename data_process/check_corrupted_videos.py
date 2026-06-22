#!/usr/bin/env python3
"""
Check for corrupted video files in Spatia data folders.

Usage:
    python -m data_process.check_corrupted_videos
    python -m data_process.check_corrupted_videos --data-root data/Spatia/frame33_fps16_2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

VIDEOS_TO_CHECK = [
    "clip.mp4",
    "train_target_rgb.mp4",
    "train_target_scene_proj_rgb.mp4",
    "train_preceding_rgb.mp4",
    "train_preceding_scene_proj_rgb.mp4",
    "train_reference_rgb.mp4",
]


def is_video_corrupted(video_path: Path) -> bool:
    """Check if a video file is corrupted."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return True
        ret, frame = cap.read()
        cap.release()
        return not ret
    except Exception:
        return True


def list_sample_folders(data_root: Path) -> list[Path]:
    """List all sample folders (8-digit numbered folders)."""
    folders = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name.isdigit() and len(p.name) == 8:
            folders.append(p)
    return folders


def main():
    parser = argparse.ArgumentParser(description="Check for corrupted video files")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/Spatia/frame33_fps16_2000",
        help="Root directory containing sample folders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for corrupted folders list (default: {data_root}/corrupted_folders.txt)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_file = (
        Path(args.output) if args.output else data_root / "corrupted_folders.txt"
    )

    print("=" * 60)
    print("Corrupted Video Checker")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Output file: {output_file}")
    print("-" * 60)

    folders = list_sample_folders(data_root)
    print(f"Found {len(folders)} sample folders")

    corrupted_folders = []

    for folder in tqdm(folders, desc="Checking"):
        folder_corrupted = []
        for video_name in VIDEOS_TO_CHECK:
            video_path = folder / video_name
            if video_path.exists() and is_video_corrupted(video_path):
                folder_corrupted.append(video_name)
        if folder_corrupted:
            corrupted_folders.append((folder.name, folder_corrupted))

    print(f"\n{'=' * 60}")
    print(f"Found {len(corrupted_folders)} folders with corrupted videos")
    print(f"{'=' * 60}")

    # Show first 20
    for folder_name, videos in corrupted_folders[:20]:
        print(f"{folder_name}: {videos}")

    if len(corrupted_folders) > 20:
        print(f"... and {len(corrupted_folders) - 20} more")

    # Save full list
    with open(output_file, "w") as f:
        for folder_name, videos in corrupted_folders:
            f.write(f"{folder_name}: {','.join(videos)}\n")

    print(f"\nFull list saved to {output_file}")

    # Also save just folder names for easy deletion
    folders_only_file = output_file.with_suffix(".folders.txt")
    with open(folders_only_file, "w") as f:
        for folder_name, _ in corrupted_folders:
            f.write(f"{folder_name}\n")
    print(f"Folder names only saved to {folders_only_file}")


if __name__ == "__main__":
    main()
