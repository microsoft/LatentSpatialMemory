#!/usr/bin/env python3
"""
Extract first frames from Spatia sample videos.

Usage:
    python scripts/extract_first_frames.py data/Spatia/demo

Input:
    A data root containing first-level sample directories. Each sample directory
    should contain train_target_rgb.mp4.

Output:
    Writes first_frame.png into each sample directory. Existing first_frame.png
    files are kept and counted as successful samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

VIDEO_NAME = "train_target_rgb.mp4"
OUTPUT_NAME = "first_frame.png"


def extract_first_frame(video_path: Path, output_path: Path) -> bool:
    """Extract the first video frame and save it as a PNG."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    try:
        ret, frame = cap.read()
    finally:
        cap.release()

    if not ret or frame is None:
        return False

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except cv2.error:
        return False

    temp_path = output_path.with_name(f".{output_path.name}.tmp")

    try:
        Image.fromarray(frame_rgb).save(temp_path, format="PNG")
        temp_path.replace(output_path)
    except OSError:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract first_frame.png from train_target_rgb.mp4 in each sample directory.",
    )
    parser.add_argument(
        "target_path",
        type=Path,
        help="Directory containing first-level sample directories.",
    )
    args = parser.parse_args()

    target_path = args.target_path
    if not target_path.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_path}")
    if not target_path.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_path}")

    sample_dirs = sorted(path for path in target_path.iterdir() if path.is_dir())
    if not sample_dirs:
        raise RuntimeError(f"No first-level sample directories found: {target_path}")

    print("=" * 60)
    print("First Frame Extractor")
    print("=" * 60)
    print(f"Target: {target_path.resolve()}")
    print(f"Sample directories: {len(sample_dirs)}")
    print(f"Input video: {VIDEO_NAME}")
    print(f"Output image: {OUTPUT_NAME}")
    print("-" * 60)

    success_count = 0
    failed_count = 0
    missing_video_count = 0
    extraction_failed_count = 0
    write_failed_count = 0

    with tqdm(total=len(sample_dirs), desc="Extracting", unit="sample") as progress:
        for sample_dir in sample_dirs:
            video_path = sample_dir / VIDEO_NAME
            output_path = sample_dir / OUTPUT_NAME

            if output_path.exists():
                success_count += 1
            elif not video_path.exists():
                missing_video_count += 1
                failed_count += 1
            else:
                try:
                    if extract_first_frame(video_path, output_path):
                        success_count += 1
                    else:
                        extraction_failed_count += 1
                        failed_count += 1
                except OSError:
                    write_failed_count += 1
                    failed_count += 1

            progress.set_postfix(
                success=success_count,
                failed=failed_count,
            )
            progress.update(1)

    print("=" * 60)
    print("Summary:")
    print(f"  Total samples: {len(sample_dirs)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Missing videos: {missing_video_count}")
    print(f"  Extraction failures: {extraction_failed_count}")
    print(f"  Write failures: {write_failed_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
