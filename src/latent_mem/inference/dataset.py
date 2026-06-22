"""
Inference dataset.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def load_frame_from_video(video_path: str, frame_idx: int) -> Image.Image:
    """Load one frame from a video and resize to target_size."""
    cap = cv2.VideoCapture(video_path)
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Could not read frame {frame_idx} from: {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    return img


class InferenceDataset(Dataset):
    def __init__(
        self,
        input_dirs: Sequence[Path | str],
        num_frames: int,
        start_frame: Optional[int] | Sequence[Optional[int]],
    ) -> None:
        self.input_dirs = [Path(d) for d in input_dirs]
        if not self.input_dirs:
            raise ValueError("input_dirs must contain at least one directory")

        self.num_frames = num_frames
        self.prompts = [None] * len(self.input_dirs)

        # Normalize start_frame to per-sample list
        if start_frame is None:
            self.start_frames = [None] * len(self.input_dirs)
        elif isinstance(start_frame, int):
            self.start_frames = [start_frame] * len(self.input_dirs)
        else:
            self.start_frames = list(start_frame)

        # Validate lengths
        if len(self.start_frames) != len(self.input_dirs):
            raise ValueError("start_frame list length must match input_dirs")
        if len(self.prompts) != len(self.input_dirs):
            raise ValueError("prompts list length must match input_dirs")

    def __len__(self) -> int:
        return len(self.input_dirs)

    def __getitem__(self, idx: int) -> dict:
        input_dir = self.input_dirs[idx]
        start_frame = self.start_frames[idx]
        prompt_override = self.prompts[idx]

        total_frames = self._get_total_frames(input_dir)
        t0 = self._resolve_start_frame(input_dir, start_frame, total_frames)
        first_frame = self._load_first_frame(input_dir, t0)
        prompt = self._load_prompt(input_dir, prompt_override)

        return {
            "data_dir": input_dir,
            "t0": t0,
            "first_frame": first_frame,
            "prompt": prompt,
            "total_frames_available": total_frames,
        }

    def _get_total_frames(self, input_dir: Path) -> Optional[int]:
        geometry_path = input_dir / "geometry.npz"
        if not geometry_path.exists():
            return None
        geometry_data = np.load(geometry_path)
        return int(geometry_data["poses_c2w"].shape[0])

    def _resolve_start_frame(
        self, input_dir: Path, start_frame: Optional[int], total_frames: Optional[int]
    ) -> int:
        # Use explicit start_frame if provided
        if start_frame is not None:
            if total_frames is not None:
                max_start = total_frames - self.num_frames
                if start_frame < 0 or start_frame > max_start:
                    raise ValueError(
                        f"start_frame {start_frame} must be between 0 and {max_start}"
                    )
            return start_frame

        # Try to load from train_sample.json
        sample_json_path = input_dir / "train_sample.json"
        if sample_json_path.exists():
            with sample_json_path.open("r", encoding="utf-8") as f:
                sample_meta = json.load(f)
            return int(sample_meta.get("t0", 0))

        return 0

    def _load_first_frame(self, input_dir: Path, t0: int) -> Image.Image:
        # Try to load from video first
        clip_path = input_dir / "clip.mp4"
        if clip_path.exists():
            return load_frame_from_video(str(clip_path), t0)

        # Try image files
        for candidate in [
            "first_frame.png",
            "frame_0.png",
            "train_target_rgb_frame0.png",
        ]:
            img_path = input_dir / candidate
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                return img

        raise FileNotFoundError(f"No first frame found in {input_dir}")

    def _load_prompt(self, input_dir: Path, prompt_override: Optional[str]) -> str:
        # Use override if provided
        if prompt_override:
            return prompt_override.strip()

        # Load from file
        prompt_path = input_dir / "clip.txt"
        if prompt_path.exists():
            with prompt_path.open("r", encoding="utf-8") as f:
                return f.read().strip()

        return ""
