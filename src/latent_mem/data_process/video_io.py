from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def list_video_files(video_dir: str | Path, recursive: bool = True) -> list[Path]:
    """List video files in a directory.

    Args:
        video_dir: Directory to search for videos.
        recursive: If True, search subdirectories recursively.

    Returns:
        Sorted list of video file paths.
    """
    root = Path(video_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)

    if recursive:
        # Recursively find all video files
        videos = []
        for ext in VIDEO_EXTS:
            videos.extend(root.rglob(f"*{ext}"))
            videos.extend(root.rglob(f"*{ext.upper()}"))
        return sorted(set(videos))
    else:
        # Only search top-level directory
        return sorted([p for p in root.iterdir() if p.suffix.lower() in VIDEO_EXTS])


def load_video_frames(
    video_path: str | Path,
    start: int = 0,
    stride: int = 1,
    max_frames: Optional[int] = None,
    target_size: Optional[int | tuple[int, int]] = None,
) -> list[np.ndarray]:
    """Load video frames with optional resize.

    Args:
        video_path: Path to video file.
        start: Start frame index.
        stride: Frame stride.
        max_frames: Maximum number of frames to load.
        target_size: Target size for resize. If int, resize to (target_size, target_size).
                    If tuple, resize to (width, height). If None, no resize.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Parse target_size
    resize_wh = None
    if target_size is not None:
        if isinstance(target_size, int):
            resize_wh = (target_size, target_size)
        else:
            resize_wh = (target_size[0], target_size[1])

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= start and (idx - start) % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize_wh is not None:
                frame = cv2.resize(frame, resize_wh, interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from video: {video_path}")
    return frames


def get_video_fps(video_path: str | Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 0:
        return 16.0
    return float(fps)


def get_video_frame_count(video_path: str | Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if count <= 0:
        return len(load_video_frames(video_path))
    return count


def load_mask_sequence(
    mask_dir: str | Path, indices: Iterable[int]
) -> list[np.ndarray]:
    import cv2

    mask_dir = Path(mask_dir)
    if not mask_dir.is_dir():
        raise NotADirectoryError(mask_dir)

    masks: list[np.ndarray] = []
    for idx in indices:
        path = mask_dir / f"{idx:06d}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        masks.append(mask > 0)
    return masks
