#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from .distributed import get_rank_info, shard_items
from .qwen3vl_prompts import (
    Qwen3VLClient,
    build_caption_prompt,
    normalize_caption,
)


def _parse_video_keys(raw: str) -> list[str]:
    if not raw:
        return []
    keys = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if item.lower().endswith(".mp4"):
            item = item[:-4]
        keys.append(item)
    return keys


def _list_video_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _log_message(
    message: str,
    rank: int,
    world_size: int,
    level: str = "INFO",
    quiet_nonzero: bool = False,
) -> None:
    if level not in {"INFO", "WARN", "ERROR"}:
        return
    if quiet_nonzero and rank != 0 and level != "ERROR":
        return
    tag = f"[Caption][{level}][R{rank}/{world_size}]"
    print(f"{tag} {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video captioning with LiteLLM multimodal API."
    )
    parser.add_argument("--input-root", type=str, default="data/train")
    parser.add_argument("--video-keys", type=str, required=True)
    parser.add_argument(
        "--qwen-model-path", type=str, default="qwen/qwen3-vl-8b-instruct"
    )
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default="DASHSCOPE_API_KEY")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--quiet-nonzero", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--caption-prompt", type=str, default=None)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-min-frames", type=int, default=4)
    parser.add_argument("--video-max-frames", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_keys = _parse_video_keys(args.video_keys)
    if not video_keys:
        raise ValueError("video_keys is empty after parsing.")

    rank, world_size, _local_rank = get_rank_info()

    client = Qwen3VLClient(
        model_path=args.qwen_model_path,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        max_new_tokens=args.max_new_tokens,
        video_fps=args.video_fps,
        video_min_frames=args.video_min_frames,
        video_max_frames=args.video_max_frames,
    )

    root = Path(args.input_root)
    video_dirs = _list_video_dirs(root)
    if args.max_videos is not None:
        video_dirs = video_dirs[: args.max_videos]

    # Shard after filtering because captioning cost comes from real API tasks.
    tasks = []
    for video_dir in video_dirs:
        for key in video_keys:
            video_path = video_dir / f"{key}.mp4"
            if not video_path.exists():
                continue
            out_path = video_dir / f"{key}.txt"
            if (
                args.skip_existing
                and out_path.exists()
                and out_path.read_text(encoding="utf-8").strip()
            ):
                continue
            tasks.append((video_dir, video_path, out_path))

    all_task_count = len(tasks)
    tasks = shard_items(tasks, rank, world_size)
    _log_message(
        f"Shard summary: dirs_total={len(video_dirs)} tasks_pending_total={all_task_count} tasks_assigned={len(tasks)}",
        rank,
        world_size,
        quiet_nonzero=False,
    )

    pbar = tqdm(total=len(tasks), disable=rank != 0, desc="captioning")
    for video_dir, video_path, out_path in tasks:
        video_id = video_dir.name
        if rank == 0:
            pbar.set_description(f"captioning {video_id}")

        try:
            prompt_spec = build_caption_prompt(
                caption_prompt=args.caption_prompt,
            )
            output_text, finish_reason = client.complete(
                input_path=str(video_path),
                prompt=prompt_spec.prompt,
            )
            caption = normalize_caption(
                output_text,
                max_words=prompt_spec.max_words,
            )
        except Exception as exc:
            _log_message(
                f"Caption failed for video={video_path} out={out_path}: {exc}",
                rank,
                world_size,
                level="ERROR",
                quiet_nonzero=args.quiet_nonzero,
            )
            raise
        out_path.write_text(caption + "\n", encoding="utf-8")
        if finish_reason == "length":
            _log_message(
                f"Caption hit length limit for video={video_path} out={out_path} finish_reason={finish_reason} max_new_tokens={args.max_new_tokens}",
                rank,
                world_size,
                level="WARN",
                quiet_nonzero=args.quiet_nonzero,
            )
        elif finish_reason != "stop":
            _log_message(
                f"Unexpected finish_reason for video={video_path} out={out_path} finish_reason={finish_reason} max_new_tokens={args.max_new_tokens}",
                rank,
                world_size,
                level="WARN",
                quiet_nonzero=args.quiet_nonzero,
            )

        if rank == 0:
            pbar.update(1)

    if rank == 0:
        pbar.close()


if __name__ == "__main__":
    main()
