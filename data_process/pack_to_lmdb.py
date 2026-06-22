#!/usr/bin/env python3
"""
Pack Spatia data folders into sharded LMDB.

Packs .pt (latents) and .txt (captions) files from each sample folder
into a sharded LMDB for efficient training data loading.

Also extracts and packs the first frame of train_target_rgb.mp4 for I2V training.

Usage:
    python -m data_process.pack_to_lmdb
    python -m data_process.pack_to_lmdb --data-root data/Spatia/frame33_fps16_2000

Output directory is automatically derived: {data_root}_lmdb
  e.g., data/Spatia/frame33_fps16_2000 -> data/Spatia/frame33_fps16_2000_lmdb
"""

from __future__ import annotations

import argparse
import io
import pickle
from pathlib import Path

import cv2
import lmdb
from PIL import Image
from tqdm import tqdm

GB = 1024**3


# Files required by SpatiaLMDBDataset at training time.
REQUIRED_FILES_TO_PACK = [
    ("train_target_rgb", ".pt"),
    ("train_target_rgb", ".txt"),
    ("train_target_scene_proj_rgb", ".pt"),
    ("train_preceding_rgb", ".pt"),
    ("train_preceding_scene_proj_rgb", ".pt"),
    ("train_sample", ".json"),
]


# Files preserved when present, but not required for a trainable sample.
OPTIONAL_FILES_TO_PACK = [
    ("train_reference_rgb", ".pt"),
    ("clip", ".txt"),
]


def get_lmdb_root_from_data_root(data_root: str) -> str:
    """
    Derive LMDB output path from data root.
    e.g., data/Spatia/frame33_fps16_2000 -> data/Spatia/frame33_fps16_2000_lmdb
    """
    data_root = data_root.rstrip("/")
    return f"{data_root}_lmdb"


def list_sample_folders(data_root: Path) -> list[Path]:
    """List all sample folders (8-digit numbered folders)."""
    folders = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name.isdigit() and len(p.name) == 8:
            folders.append(p)
    return folders


def extract_first_frame(video_path: Path) -> bytes | None:
    """
    Extract the first frame from a video file and return as PNG bytes.
    Returns None if extraction fails.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Encode as PNG bytes
    img = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def read_sample_data(
    folder: Path,
    required_files_to_pack: list[tuple[str, str]],
    optional_files_to_pack: list[tuple[str, str]],
    extract_first_frames: list[str] | None = None,
    skip_stats: dict[str, int] | None = None,
) -> dict[str, bytes] | None:
    """
    Read all files for one sample folder.
    Returns dict of {filename: bytes} or None if the sample is not trainable.

    Args:
        folder: Sample folder path
        required_files_to_pack: List of (stem, extension) tuples that must exist
        optional_files_to_pack: List of (stem, extension) tuples preserved if present
        extract_first_frames: List of video stems to extract first frame from
                              (e.g., ["train_target_rgb"] -> extracts first frame
                               and saves as "train_target_rgb_frame0.png")
        skip_stats: Optional counter for skipped sample reasons.
    """
    data: dict[str, bytes] = {}

    # Reject samples missing files that the training dataset reads directly.
    for stem, ext in required_files_to_pack:
        file_path = folder / f"{stem}{ext}"
        if not file_path.exists():
            if skip_stats is not None:
                skip_stats["skipped_missing_required"] += 1
            return None

    # Read train files first, then preserve optional extras when available.
    for stem, ext in required_files_to_pack + optional_files_to_pack:
        file_path = folder / f"{stem}{ext}"
        filename = f"{stem}{ext}"

        if file_path.exists():
            with open(file_path, "rb") as f:
                data[filename] = f.read()

    # Require first-frame extraction so LMDB matches the training contract.
    if extract_first_frames:
        for video_stem in extract_first_frames:
            video_path = folder / f"{video_stem}.mp4"
            if not video_path.exists():
                if skip_stats is not None:
                    skip_stats["skipped_missing_video"] += 1
                return None

            frame_bytes = extract_first_frame(video_path)
            if frame_bytes is None:
                if skip_stats is not None:
                    skip_stats["skipped_bad_video"] += 1
                return None

            data[f"{video_stem}_frame0.png"] = frame_bytes

    return data


def estimate_sample_size(
    folders: list[Path],
    required_files_to_pack: list[tuple[str, str]],
    optional_files_to_pack: list[tuple[str, str]],
    num_samples: int = 10,
) -> int:
    """Estimate average sample size by checking a few samples."""
    import random

    sample_folders = random.sample(folders, min(num_samples, len(folders)))
    total_size = 0
    count = 0

    for folder in sample_folders:
        for stem, ext in required_files_to_pack + optional_files_to_pack:
            file_path = folder / f"{stem}{ext}"
            if file_path.exists():
                total_size += file_path.stat().st_size
        count += 1

    return int(total_size / count) if count > 0 else 0


def safe_put_batch(
    env: lmdb.Environment,
    buffer: list[tuple[bytes, bytes]],
    grow_factor: float = 1.5,
) -> None:
    """Write a batch to LMDB, growing map size if needed."""
    while True:
        try:
            with env.begin(write=True) as txn:
                for k, v in buffer:
                    inserted = txn.put(k, v, overwrite=False)
                    if not inserted:
                        key = k.decode("utf-8", errors="replace")
                        raise RuntimeError(f"Duplicate LMDB key found: {key}")
            return
        except lmdb.MapFullError:
            current = env.info().get("map_size", 0)
            new_size = int(current * grow_factor) if current else int(1.5 * GB)
            env.set_mapsize(new_size)
            print(f"  LMDB map full; expanded to {new_size / GB:.2f} GB")


def build_sharded_lmdb(
    data_root: str,
    lmdb_root: str,
    required_files_to_pack: list[tuple[str, str]],
    optional_files_to_pack: list[tuple[str, str]],
    extract_first_frames: list[str] | None = None,
    target_shard_size_gb: float = 10.0,
    write_batch: int = 256,
):
    """
    Build sharded LMDB from Spatia data folders.

    Args:
        data_root: Root directory containing sample folders (00000000, 00000001, ...)
        lmdb_root: Output LMDB directory
        required_files_to_pack: List of (stem, extension) tuples that must exist
        optional_files_to_pack: List of (stem, extension) tuples preserved if present
        extract_first_frames: List of video stems to extract first frame from
        target_shard_size_gb: Target size per shard in GB
        write_batch: Samples per transaction
    """
    data_root = Path(data_root).resolve()
    lmdb_root = Path(lmdb_root).resolve()

    print(f"Data root: {data_root}")
    print(f"LMDB root: {lmdb_root}")

    if lmdb_root.exists():
        raise FileExistsError(
            f"LMDB output directory already exists: {lmdb_root}. "
            "Delete it manually or choose a different --lmdb-root before packing."
        )

    # List sample folders
    print("Listing sample folders...")
    folders = list_sample_folders(data_root)
    print(f"Found {len(folders)} sample folders")

    if not folders:
        raise RuntimeError(f"No sample folders found in {data_root}")

    skip_stats = {
        "skipped_missing_required": 0,
        "skipped_missing_video": 0,
        "skipped_bad_video": 0,
    }
    written_samples = 0

    # Estimate sample size
    print("Estimating sample sizes...")
    avg_size = estimate_sample_size(
        folders,
        required_files_to_pack,
        optional_files_to_pack,
    )
    print(f"Average sample size: {avg_size / 1024**2:.2f} MB")

    # Calculate shard size
    target_bytes = target_shard_size_gb * GB
    shard_size = max(1, int(target_bytes / avg_size)) if avg_size > 0 else 1000
    num_shards = (len(folders) + shard_size - 1) // shard_size

    print("Shard configuration:")
    print(f"  Target shard size: {target_shard_size_gb} GB")
    print(f"  Samples per shard: {shard_size}")
    print(f"  Total shards: {num_shards}")

    # Create output directory
    lmdb_root.mkdir(parents=True, exist_ok=True)

    # Build shards
    folder_idx = 0
    for shard_id in range(num_shards):
        shard_folders = folders[folder_idx : folder_idx + shard_size]
        folder_idx += shard_size

        if not shard_folders:
            break

        shard_dir = lmdb_root / f"shard_{shard_id:03d}.lmdb"
        shard_dir.mkdir(parents=True, exist_ok=True)

        # Estimate map size for this shard
        est_bytes = avg_size * len(shard_folders)
        map_size = max(int(est_bytes * 3), int(est_bytes + 2 * GB), 1 << 30)

        print(f"{'=' * 60}")
        print(f"Building shard {shard_id}/{num_shards - 1}: {shard_dir.name}")
        print(f"  Samples: {len(shard_folders)}")
        print(f"  Est. size: {est_bytes / GB:.2f} GB")
        print(f"  Map size: {map_size / GB:.2f} GB")

        env = lmdb.open(
            str(shard_dir),
            map_size=map_size,
            subdir=True,
            readonly=False,
            lock=True,
            readahead=False,
            meminit=False,
            sync=True,
            max_readers=256,
        )

        buffer = []
        shard_written_samples = 0
        for folder in tqdm(shard_folders, desc=f"Shard {shard_id}"):
            data = read_sample_data(
                folder,
                required_files_to_pack,
                optional_files_to_pack,
                extract_first_frames,
                skip_stats,
            )
            if data is None:
                continue

            key = folder.name.encode("utf-8")
            value = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            buffer.append((key, value))
            shard_written_samples += 1
            written_samples += 1

            if len(buffer) >= write_batch:
                safe_put_batch(env, buffer)
                buffer.clear()

        if buffer:
            safe_put_batch(env, buffer)
            buffer.clear()

        env.sync()
        env.close()
        print(f"  Valid files in shard: {shard_written_samples}/{len(shard_folders)}")
        print(f"  Finished shard {shard_id}")

    skipped_samples = sum(skip_stats.values())
    print("Packing summary:")
    print(f"  Written samples: {written_samples}/{len(folders)}")
    print(f"  Skipped samples: {skipped_samples}")
    print(f"    Missing required files: {skip_stats['skipped_missing_required']}")
    print(f"    Missing first-frame video: {skip_stats['skipped_missing_video']}")
    print(f"    Failed first-frame extraction: {skip_stats['skipped_bad_video']}")

    if written_samples == 0:
        raise RuntimeError("No valid samples written")

    print(f"{'=' * 60}")
    print("All shards complete!")
    print(f"Output: {lmdb_root}")
    print(f"{'=' * 60}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack Spatia data to sharded LMDB")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/Spatia/frame33_fps16_2000",
        help="Root directory containing sample folders",
    )
    parser.add_argument(
        "--lmdb-root",
        type=str,
        default=None,
        help="Output LMDB directory (default: {data_root}_lmdb)",
    )
    parser.add_argument(
        "--target-shard-size-gb",
        type=float,
        default=10.0,
        help="Target size per shard in GB",
    )
    parser.add_argument(
        "--write-batch",
        type=int,
        default=256,
        help="Samples per transaction",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = args.data_root
    lmdb_root = args.lmdb_root or get_lmdb_root_from_data_root(data_root)

    # Videos to extract first frame from (for I2V training)
    extract_first_frames = ["train_target_rgb"]

    print("=" * 60)
    print("Spatia LMDB Packer")
    print("=" * 60)
    print("Configuration:")
    print(f"  Data root: {data_root}")
    print(f"  LMDB root: {lmdb_root}")
    print(f"  Target shard size: {args.target_shard_size_gb} GB")
    print(f"  Required files: {[f'{s}{e}' for s, e in REQUIRED_FILES_TO_PACK]}")
    print(f"  Optional files: {[f'{s}{e}' for s, e in OPTIONAL_FILES_TO_PACK]}")
    print(
        f"  Extract first frames: {[f'{s}.mp4 -> {s}_frame0.png' for s in extract_first_frames]}"
    )
    print("-" * 60)

    build_sharded_lmdb(
        data_root=data_root,
        lmdb_root=lmdb_root,
        required_files_to_pack=REQUIRED_FILES_TO_PACK,
        optional_files_to_pack=OPTIONAL_FILES_TO_PACK,
        extract_first_frames=extract_first_frames,
        target_shard_size_gb=args.target_shard_size_gb,
        write_batch=args.write_batch,
    )


if __name__ == "__main__":
    main()
