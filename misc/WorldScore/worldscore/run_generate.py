# If you haven't run
# "export $(grep -v '^#' .env | xargs)"
# in the shell, please run it first!

from argparse import ArgumentParser, Namespace
from typing import Optional

from PIL import Image

from worldscore.benchmark.utils.utils import check_model, merge_video
from worldscore.common.utils import print_banner


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="WorldScore Generation Tool")
    # specify the domain
    parser.add_argument(
        "--model_name",
        type=str,
        default="wonderjourney",
        help="model name to evaluate",
    )
    parser.add_argument(
        "--visual_movement",
        type=str,
        default="static",
        choices=["static", "dynamic"],
        help="type of visual movement",
    )
    parser.add_argument(
        "--merge_frames",
        "-mf",
        action="store_true",
        help="merge frames into videos",
    )

    return parser


def run_merge_frames(args: Namespace) -> None:
    import os
    from pathlib import Path

    from omegaconf import OmegaConf

    print("-- Model name: ", args.model_name)
    print("-- Merging frames into videos...")
    print(
        "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
    )
    base_config = OmegaConf.load(os.path.join("config/base_config.yaml"))
    try:
        config = OmegaConf.load(
            os.path.join("config/model_configs", f"{args.model_name}.yaml")
        )
    except FileNotFoundError:
        print(f"-- Model config file not found for {args.model_name}")
        return
    config = OmegaConf.merge(base_config, config)
    config.visual_movement = args.visual_movement
    # Interpolate environment variables in the YAML file
    config = OmegaConf.to_container(config, resolve=True)

    runs_root = config["runs_root"]
    output_dir = config["output_dir"]
    root_path = Path(
        runs_root,
        output_dir,
        config["visual_movement"],
    )

    count = 0
    total_count = 0
    if config["visual_movement"] == "static":
        visual_styles = sorted([x.name for x in root_path.iterdir() if x.is_dir()])

        for visual_style in visual_styles:
            visual_style_dir = root_path / visual_style
            scene_types = sorted(
                [x.name for x in visual_style_dir.iterdir() if x.is_dir()]
            )
            for scene_type in scene_types:
                scene_type_dir = visual_style_dir / scene_type

                category_list = sorted(
                    [f.name for f in scene_type_dir.iterdir() if f.is_dir()]
                )
                for category in category_list:
                    category_dir = scene_type_dir / category
                    instance_list = sorted(
                        [f.name for f in category_dir.iterdir() if f.is_dir()]
                    )
                    for instance in instance_list:
                        total_count += 1
                        instance_dir = category_dir / instance

                        frame_dir = instance_dir / "frames"

                        if not frame_dir.exists() or not frame_dir.is_dir():
                            continue

                        frames_files = sorted(
                            [
                                frame_dir / x.name
                                for x in frame_dir.iterdir()
                                if x.is_file()
                                and x.name.lower().endswith((".png", ".jpg"))
                            ]
                        )
                        if len(frames_files) == 0:
                            continue

                        frames = [Image.open(f) for f in frames_files]
                        print(
                            f"-- {count + 1} / {total_count} Merging {len(frames)} frames, fps={config.get('fps', 10)}"
                        )
                        print(f"-- {instance_dir}/videos")
                        print(
                            "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
                        )
                        merge_video(
                            frames,
                            save_dir=f"{instance_dir}/videos",
                            fps=config.get("fps", 10),
                        )
                        count += 1

    elif config["visual_movement"] == "dynamic":
        visual_styles = sorted([x.name for x in root_path.iterdir() if x.is_dir()])

        for visual_style in visual_styles:
            visual_style_dir = root_path / visual_style
            motion_types = sorted(
                [x.name for x in visual_style_dir.iterdir() if x.is_dir()]
            )
            for motion_type in motion_types:
                motion_type_dir = visual_style_dir / motion_type

                instance_list = sorted(
                    [f.name for f in motion_type_dir.iterdir() if f.is_dir()]
                )
                for instance in instance_list:
                    total_count += 1
                    instance_dir = motion_type_dir / instance

                    frame_dir = instance_dir / "frames"

                    if not frame_dir.exists() or not frame_dir.is_dir():
                        continue

                    frames_files = sorted(
                        [
                            frame_dir / x.name
                            for x in frame_dir.iterdir()
                            if x.is_file() and x.name.lower().endswith((".png", ".jpg"))
                        ]
                    )
                    if len(frames_files) == 0:
                        continue

                    frames = [Image.open(f) for f in frames_files]
                    print(
                        f"-- {count + 1} / {total_count} Merging {len(frames)} frames, fps={config.get('fps', 10)}"
                    )
                    print(f"-- {instance_dir}/videos")
                    merge_video(
                        frames,
                        save_dir=f"{instance_dir}/videos",
                        fps=config.get("fps", 10),
                    )
                    print(
                        "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
                    )
                    count += 1
    else:
        raise ValueError(f"Invalid visual movement: {config['visual_movement']}")

    print(
        f"-- {args.model_name} {args.visual_movement} Merged: {count} / {total_count} data points"
    )
    print(
        "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
    )


def main(argv: Optional[list] = None) -> None:
    import sys

    parser = get_parser()

    print_banner("GENERATION")

    if argv is None:
        argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        parser.print_help()
        return

    args = parser.parse_args(argv)

    if "--merge_frames" in argv or "-mf" in argv:
        run_merge_frames(args)
        return

    assert check_model(args.model_name), "Model not exists!"
    # run_generate(args)


if __name__ == "__main__":
    main()
