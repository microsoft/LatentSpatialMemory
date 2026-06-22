#!/usr/bin/env python3
"""Pack a directory into sample-count tar shards."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack a directory into tar shards by first-level entry count.",
    )
    parser.add_argument(
        "target_path",
        type=Path,
        help="Directory to pack.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Write compressed .tar.gz shards.",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove source files after each shard is created and verified.",
    )
    parser.add_argument(
        "--per-shard",
        type=int,
        default=1000,
        help="Number of first-level entries per shard (default: 1000).",
    )
    return parser.parse_args()


def prepare_output_dir(target_path: Path) -> Path:
    output_dir = target_path.parent / f"{target_path.name}_tar_shards"
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Output path exists but is not a directory: {output_dir}"
            )
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_shard_path(
    output_dir: Path,
    target_name: str,
    shard_index: int,
    use_zip: bool,
) -> Path:
    suffix = ".tar.gz" if use_zip else ".tar"
    return output_dir / f"{target_name}-{shard_index:05d}{suffix}"


def pack_target(
    target_path: Path,
    use_zip: bool,
    remove_sources: bool,
    per_shard: int,
) -> None:
    if not target_path.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_path}")
    if not target_path.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_path}")
    if not target_path.name:
        raise ValueError(f"Target path must have a directory name: {target_path}")
    if per_shard < 1:
        raise ValueError(f"--per-shard must be at least 1, got {per_shard}")

    target_path = target_path.resolve()
    entries = sorted(target_path.iterdir(), key=lambda path: path.name)
    if not entries:
        raise RuntimeError(
            f"No first-level files or directories found under target: {target_path}"
        )

    shards = [
        entries[start : start + per_shard]
        for start in range(0, len(entries), per_shard)
    ]
    output_dir = prepare_output_dir(target_path)
    mode = "w:gz" if use_zip else "w"

    print(f"Target: {target_path}")
    print(f"Output: {output_dir}")
    print(f"First-level entries: {len(entries)}")
    print(f"Entries per shard: {per_shard}")
    print(f"Shards: {len(shards)}")
    print(f"Format: {'tar.gz' if use_zip else 'tar'}")
    print(f"Remove sources: {remove_sources}")

    with tqdm(total=len(entries), desc="Packing", unit="entry") as progress:
        for shard_index, shard_entries in enumerate(shards):
            final_path = make_shard_path(
                output_dir, target_path.name, shard_index, use_zip
            )
            temp_path = final_path.with_name(f".{final_path.name}.tmp")
            tqdm.write(
                f"Writing shard {shard_index + 1}/{len(shards)}: {final_path.name}"
            )

            # Write a temporary archive first so failed shards never look complete.
            try:
                with tarfile.open(temp_path, mode, format=tarfile.GNU_FORMAT) as tar:
                    for entry in shard_entries:
                        arcname = (Path(target_path.name) / entry.name).as_posix()
                        tar.add(entry, arcname=arcname, recursive=True)
                        progress.update(1)

                if not temp_path.is_file():
                    raise RuntimeError(f"Shard was not created: {temp_path}")
                temp_path.rename(final_path)
            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise

            if remove_sources:
                for entry in shard_entries:
                    entry_root = entry.resolve()
                    if entry_root == target_path or not entry_root.is_relative_to(
                        target_path
                    ):
                        raise RuntimeError(
                            f"Refusing to remove path outside target root: {entry}"
                        )

                    if entry.is_symlink() or entry.is_file():
                        entry.unlink()
                    elif entry.is_dir():
                        shutil.rmtree(entry)

    print("Done.")


def main() -> None:
    args = parse_args()
    pack_target(
        target_path=args.target_path,
        use_zip=args.zip,
        remove_sources=args.remove,
        per_shard=args.per_shard,
    )


if __name__ == "__main__":
    main()
