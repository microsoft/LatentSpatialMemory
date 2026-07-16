from __future__ import annotations

import hashlib
import importlib
import random
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2

CLIP_FILENAME = "clip.mp4"
SOURCE_VIDEO_PATH_FILENAME = "source_video_path.txt"
CLIP_FPS_TOLERANCE = 0.1


@dataclass(frozen=True)
class ClipExtractionSpec:
    """Deterministic clip extraction parameters derived from one source video."""

    start_time_sec: float
    clip_seconds: float
    source_fps: float
    source_frame_count: int
    source_width: int
    source_height: int


def get_video_info(video_path: Path) -> tuple[float, int, int, int] | None:
    """Return basic video metadata as (fps, frames, width, height)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        return None
    return fps, frame_count, width, height


def make_clip_rng(seed: int, video_path: Path) -> random.Random:
    """Create a deterministic RNG from the seed and video path."""
    key = f"{seed}:{video_path.as_posix()}".encode("utf-8")
    digest = hashlib.md5(key).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    return random.Random(seed_int)


def normalize_source_video_path(video_path: str | Path) -> str:
    """Normalize a source video path for stable de-duplication."""
    return Path(video_path).resolve().as_posix()


@lru_cache(maxsize=1)
def resolve_ffmpeg_executable() -> str:
    """Resolve an ffmpeg executable from PATH or the imageio bundle."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg executable not found in PATH and imageio_ffmpeg is unavailable"
        ) from exc

    ffmpeg_path = str(imageio_ffmpeg.get_ffmpeg_exe()).strip()
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg executable not found in PATH and imageio_ffmpeg returned an empty path"
        )
    return ffmpeg_path


def check_ffmpeg_readiness() -> tuple[bool, str]:
    """Return whether clip extraction can access a usable ffmpeg binary."""
    try:
        ffmpeg_path = resolve_ffmpeg_executable()
    except RuntimeError as exc:
        return False, str(exc)
    return True, ffmpeg_path


def build_clip_extraction_spec(
    video_path: Path,
    num_frames: int,
    target_fps: int,
    seed: int,
) -> tuple[ClipExtractionSpec | None, str]:
    """Validate a source video and derive a deterministic extraction window."""
    info = get_video_info(video_path)
    if info is None:
        return None, "invalid video metadata"

    src_fps, frame_count, width, height = info
    if src_fps + 1e-6 < target_fps:
        return (
            None,
            f"video fps too low (src_fps={src_fps:.3f}, target_fps={target_fps})",
        )

    clip_seconds = num_frames / float(target_fps)
    duration = frame_count / float(src_fps)
    if duration + 1e-6 < clip_seconds:
        return None, (
            f"video too short (duration={duration:.3f}s, "
            f"required={clip_seconds:.3f}s, frames={frame_count})"
        )

    rng = make_clip_rng(seed, video_path)
    max_start = max(0.0, duration - clip_seconds)
    start_time = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
    return (
        ClipExtractionSpec(
            start_time_sec=start_time,
            clip_seconds=clip_seconds,
            source_fps=src_fps,
            source_frame_count=frame_count,
            source_width=width,
            source_height=height,
        ),
        "ok",
    )


def validate_source_dimensions(
    source_width: int,
    source_height: int,
    target_wh: tuple[int, int],
) -> tuple[bool, str]:
    """Reject videos that cannot support the requested center crop."""
    target_w, target_h = target_wh
    if source_width < target_w or source_height < target_h:
        return (
            False,
            "video too small for crop "
            f"(source={(source_width, source_height)}, target={(target_w, target_h)})",
        )
    return True, "ok"


def build_ffmpeg_extract_command(
    ffmpeg_executable: str,
    video_path: Path,
    output_path: Path,
    spec: ClipExtractionSpec,
    num_frames: int,
    target_fps: int,
    target_wh: tuple[int, int],
) -> list[str]:
    """Build the ffmpeg command used for deterministic clip extraction."""
    target_w, target_h = target_wh
    video_filter = (
        f"crop={target_w}:{target_h}:(iw-{target_w})/2:(ih-{target_h})/2,"
        f"fps={target_fps}"
    )
    return [
        ffmpeg_executable,
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{spec.start_time_sec:.6f}",
        "-i",
        str(video_path),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        video_filter,
        "-frames:v",
        str(num_frames),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]


def check_clip_file_complete(
    clip_path: Path,
    num_frames: int,
    target_fps: int,
    target_wh: tuple[int, int],
) -> bool:
    """Check whether a clip file matches the expected shape and FPS."""
    if not clip_path.exists():
        return False
    info = get_video_info(clip_path)
    if info is None:
        return False

    fps, frame_count, width, height = info
    target_w, target_h = target_wh
    return (
        frame_count == num_frames
        and width == target_w
        and height == target_h
        and abs(fps - float(target_fps)) <= CLIP_FPS_TOLERANCE
    )


def extract_clip(
    video_path: Path,
    output_path: Path,
    num_frames: int,
    target_fps: int,
    target_wh: tuple[int, int],
    seed: int,
) -> tuple[bool, str]:
    """Extract one clip from the source video with ffmpeg."""
    spec, message = build_clip_extraction_spec(
        video_path=video_path,
        num_frames=num_frames,
        target_fps=target_fps,
        seed=seed,
    )
    if spec is None:
        return False, message

    is_valid_size, size_message = validate_source_dimensions(
        source_width=spec.source_width,
        source_height=spec.source_height,
        target_wh=target_wh,
    )
    if not is_valid_size:
        return False, size_message

    try:
        ffmpeg_executable = resolve_ffmpeg_executable()
    except RuntimeError as exc:
        return False, str(exc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    temp_output_path.unlink(missing_ok=True)

    command = build_ffmpeg_extract_command(
        ffmpeg_executable=ffmpeg_executable,
        video_path=video_path,
        output_path=temp_output_path,
        spec=spec,
        num_frames=num_frames,
        target_fps=target_fps,
        target_wh=target_wh,
    )
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        temp_output_path.unlink(missing_ok=True)
        return False, f"ffmpeg invocation failed: {exc}"

    if result.returncode != 0:
        temp_output_path.unlink(missing_ok=True)
        error_text = result.stderr.strip() or result.stdout.strip()
        if not error_text:
            error_text = f"exit code {result.returncode}"
        return False, f"ffmpeg execution failed: {error_text}"

    if not check_clip_file_complete(
        clip_path=temp_output_path,
        num_frames=num_frames,
        target_fps=target_fps,
        target_wh=target_wh,
    ):
        temp_output_path.unlink(missing_ok=True)
        return False, "ffmpeg wrote an incomplete clip"

    output_path.unlink(missing_ok=True)
    temp_output_path.replace(output_path)
    return True, "ok"


def save_source_video_path(output_dir: Path, source_video_path: Path) -> None:
    """Persist the original source video path for one output folder."""
    source_path_file = output_dir / SOURCE_VIDEO_PATH_FILENAME
    source_path = normalize_source_video_path(source_video_path)
    source_path_file.write_text(f"{source_path}\n", encoding="utf-8")


def load_source_video_path(output_dir: Path) -> str | None:
    """Load the original source video path for one output folder."""
    source_path_file = output_dir / SOURCE_VIDEO_PATH_FILENAME
    if not source_path_file.exists():
        return None
    source_path = source_path_file.read_text(encoding="utf-8").strip()
    return source_path or None


def check_clip_complete(folder_path: Path, cfg) -> bool:
    """Check whether clip.mp4 exists and matches the clip config."""
    clip_path = folder_path / CLIP_FILENAME
    target_wh = (cfg.clip_target_width, cfg.clip_target_height)
    return check_clip_file_complete(
        clip_path=clip_path,
        num_frames=cfg.clip_num_frames,
        target_fps=cfg.clip_target_fps,
        target_wh=target_wh,
    )


def prepare_folder_ids(max_videos: int) -> list[str]:
    """Return standard output folder names."""
    return [f"{idx:08d}" for idx in range(max_videos)]


def shuffle_videos(all_videos: list[Path], shuffle_seed: int | None) -> list[Path]:
    """Shuffle source videos with a deterministic seed."""
    if shuffle_seed is None:
        return list(all_videos)
    rng = random.Random(shuffle_seed)
    shuffled_videos = list(all_videos)
    rng.shuffle(shuffled_videos)
    return shuffled_videos
