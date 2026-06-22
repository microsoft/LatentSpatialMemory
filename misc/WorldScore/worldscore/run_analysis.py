# If you haven't run
# "export $(grep -v '^#' .env | xargs)"
# in the shell, please run it first!

from argparse import ArgumentParser, Namespace
from typing import Optional

from worldscore.benchmark.utils.utils import check_model, get_model2type, type2model
from worldscore.common.utils import print_banner


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="WorldScore Analysis Tool")
    # specify the domain
    parser.add_argument(
        "--model_name",
        type=str,
        default="wonderjourney",
        help="model name to analyze",
    )
    parser.add_argument(
        "--json_file",
        default=None,
        help="json file name",
    )
    parser.add_argument(
        "--noise",
        "-n",
        action="store_true",
        help="use random gaussian noisy images",
    )
    parser.add_argument(
        "--noise_type",
        "-nt",
        type=str,
        default="simple",
        choices=["simple", "comparison"],
        help="noise type",
    )
    parser.add_argument(
        "--check_data",
        "-cd",
        action="store_true",
        help="check if the generation data is complete",
    )
    parser.add_argument(
        "--check_score",
        "-cs",
        action="store_true",
        help="check if the evaluation score is complete",
    )

    return parser


def run_check_data(args: Namespace) -> None:
    import os
    from pathlib import Path

    from omegaconf import OmegaConf

    from worldscore.benchmark.helpers.evaluator import Evaluator

    print("-- Model name: ", args.model_name)
    print("-- Checking data completeness...")
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

    evaluator = Evaluator(config)
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
        visual_styles = [x.name for x in root_path.iterdir() if x.is_dir()]

        for visual_style in visual_styles:
            visual_style_dir = root_path / visual_style
            scene_types = [x.name for x in visual_style_dir.iterdir() if x.is_dir()]
            for scene_type in scene_types:
                scene_type_dir = visual_style_dir / scene_type

                category_list = [f.name for f in scene_type_dir.iterdir() if f.is_dir()]
                for category in category_list:
                    category_dir = scene_type_dir / category
                    instance_list = [
                        f.name for f in category_dir.iterdir() if f.is_dir()
                    ]
                    for instance in instance_list:
                        instance_dir = category_dir / instance

                        # check if the data is complete
                        if evaluator.data_exists(instance_dir):
                            count += 1
                        else:
                            print(f"-- Data is incomplete for {instance_dir}")
                        total_count += 1
    elif config["visual_movement"] == "dynamic":
        visual_styles = [x.name for x in root_path.iterdir() if x.is_dir()]

        for visual_style in visual_styles:
            visual_style_dir = root_path / visual_style
            motion_types = [x.name for x in visual_style_dir.iterdir() if x.is_dir()]
            for motion_type in motion_types:
                motion_type_dir = visual_style_dir / motion_type

                instance_list = [
                    f.name for f in motion_type_dir.iterdir() if f.is_dir()
                ]
                for instance in instance_list:
                    instance_dir = motion_type_dir / instance

                    # check if the data is complete
                    if evaluator.data_exists(instance_dir):
                        count += 1
                    else:
                        print(f"-- Data is incomplete for {instance_dir}")
                    total_count += 1
    else:
        raise ValueError(f"Invalid visual movement: {config['visual_movement']}")

    total_count = 2000 if config["visual_movement"] == "static" else 1000
    print(
        f"-- {args.model_name} {args.visual_movement} Checked: {count} / {total_count} data points"
    )
    if count == total_count:
        print("-- Data is complete!")
        return True, count, total_count
    else:
        print("-- Data is incomplete!")
        return False, count, total_count
    print(
        "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
    )


def run_check_score(args: Namespace) -> None:
    import os
    from pathlib import Path

    from omegaconf import OmegaConf

    print("-- Model name: ", args.model_name)
    print("-- Checking score completeness...")
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
        visual_styles = [x.name for x in root_path.iterdir() if x.is_dir()]

        for visual_style in visual_styles:
            visual_style_dir = root_path / visual_style
            scene_types = [x.name for x in visual_style_dir.iterdir() if x.is_dir()]
            for scene_type in scene_types:
                scene_type_dir = visual_style_dir / scene_type

                category_list = [f.name for f in scene_type_dir.iterdir() if f.is_dir()]
                for category in category_list:
                    category_dir = scene_type_dir / category
                    instance_list = [
                        f.name for f in category_dir.iterdir() if f.is_dir()
                    ]
                    for instance in instance_list:
                        instance_dir = category_dir / instance

                        # check if the score data exists
                        score_path = instance_dir / "evaluation.json"
                        if os.path.exists(score_path):
                            count += 1
                        else:
                            print(f"-- Score data is incomplete for {instance_dir}")
                        total_count += 1
    elif config["visual_movement"] == "dynamic":
        visual_styles = [x.name for x in root_path.iterdir() if x.is_dir()]

        for visual_style in visual_styles:
            visual_style_dir = root_path / visual_style
            motion_types = [x.name for x in visual_style_dir.iterdir() if x.is_dir()]
            for motion_type in motion_types:
                motion_type_dir = visual_style_dir / motion_type

                instance_list = [
                    f.name for f in motion_type_dir.iterdir() if f.is_dir()
                ]
                for instance in instance_list:
                    instance_dir = motion_type_dir / instance

                    # check if the score data exists
                    score_path = instance_dir / "evaluation.json"
                    if os.path.exists(score_path):
                        count += 1
                    else:
                        print(f"-- Score data is incomplete for {instance_dir}")
                    total_count += 1
    else:
        raise ValueError(f"Invalid visual movement: {config['visual_movement']}")

    total_count = 2000 if config["visual_movement"] == "static" else 1000
    print(
        f"-- {args.model_name} {args.visual_movement} Checked: {count} / {total_count} data points"
    )
    if count == total_count:
        print("-- Score data is complete!")
        return True, count, total_count
    else:
        print("-- Score data is incomplete!")
        return False, count, total_count
    print(
        "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
    )


def run_analysis(args: Namespace) -> None:
    import os

    from omegaconf import OmegaConf

    from worldscore.benchmark.helpers import GetDataloader
    from worldscore.benchmark.helpers.evaluator import Analysis

    config = OmegaConf.load(os.path.join("config/base_config.yaml"))
    config.visual_movement = args.visual_movement
    config.noise = args.noise
    config.noise_type = args.noise_type
    # Interpolate environment variables in the YAML file
    config = OmegaConf.to_container(config, resolve=True)

    ### dataloader
    dataloader = GetDataloader(
        args.visual_movement, args.json_file, args.noise, args.noise_type
    )
    ###

    analysis = Analysis(config, dataloader)
    analysis.analyze()


def main(argv: Optional[list] = None) -> None:
    import sys

    parser = get_parser()

    print_banner("ANALYSIS")

    if argv is None:
        argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        parser.print_help()
        return

    args = parser.parse_args(argv)

    assert check_model(args.model_name), "Model not exists!"
    model_type = get_model2type(type2model)[args.model_name]
    if model_type == "threedgen":
        visual_movement_list = ["static"]
    else:
        visual_movement_list = ["static", "dynamic"]

    complete = True
    data_dict = {}
    for visual_movement in visual_movement_list:
        args.visual_movement = visual_movement

        if args.check_data:
            complete_, count_, total_count_ = run_check_data(args)
            complete = complete_ and complete
            data_dict[visual_movement] = (count_, total_count_)
            continue
        if args.check_score:
            complete_, count_, total_count_ = run_check_score(args)
            complete = complete_ and complete
            data_dict[visual_movement] = (count_, total_count_)
            continue

        # run_analysis(args)

    if args.check_data:
        if complete:
            print("World generation data are complete!")
        else:
            print("World generation data are incomplete!")
            for visual_movement, (count, total_count) in data_dict.items():
                print(f"{visual_movement}: {count} / {total_count}")
    if args.check_score:
        if complete:
            print("World score data are complete!")
        else:
            print("World score data are incomplete!")
            for visual_movement, (count, total_count) in data_dict.items():
                print(f"{visual_movement}: {count} / {total_count}")


if __name__ == "__main__":
    main()
