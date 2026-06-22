# If you haven't run
# "export $(grep -v '^#' .env | xargs)"
# in the shell, please run it first!

import json
import os
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
from omegaconf import OmegaConf

from worldscore.benchmark.utils.utils import check_model, get_model2type, type2model
from worldscore.common.utils import print_banner

worldscore_list = {
    "static": [
        "camera_control",
        "object_control",
        "content_alignment",
        "3d_consistency",
        "photometric_consistency",
        "style_consistency",
        "subjective_quality",
    ],
    "dynamic": [
        "motion_accuracy",
        "motion_magnitude",
        "motion_smoothness",
    ],
}


def calculate_mean_scores(
    metrics_results, visual_movement, scores, calculate_raw_score: bool = False
):
    """
    Computes the average WorldScore.

    Returns:
        tuple:
            scores: aggregates score for each metric (over all instances).
    """

    aspect_list = worldscore_list[visual_movement]

    for aspect in metrics_results:
        if aspect not in aspect_list:
            continue
        metric_score_list = []
        for metric_name, metric_scores in metrics_results[aspect].items():
            if calculate_raw_score:
                metric_score_list.append(
                    np.mean(
                        np.array(
                            [metric_score["score"] for metric_score in metric_scores]
                        ),
                        axis=0,
                    ).tolist()
                )
                scores[aspect] = metric_score_list
            else:
                metric_score_list.append(
                    np.mean(
                        np.array(
                            [
                                metric_score["score_normalized"]
                                for metric_score in metric_scores
                            ]
                        ),
                        axis=0,
                    ).item()
                )
                scores[aspect] = round(np.mean(metric_score_list).item() * 100, 2)

    return scores


def calculate_worldscore(
    model_name: str, visual_movement_list: list[str], calculate_raw_score: bool = False
) -> float:
    base_config = OmegaConf.load(os.path.join("config/base_config.yaml"))
    try:
        config = OmegaConf.load(
            os.path.join("config/model_configs", f"{model_name}.yaml")
        )
    except FileNotFoundError:
        print(f"-- Model config file not found for {model_name}")
        return
    config = OmegaConf.merge(base_config, config)
    # Interpolate environment variables in the YAML file
    config = OmegaConf.to_container(config, resolve=True)

    scores = {
        aspect: 0
        for movement in worldscore_list.keys()
        for aspect in worldscore_list[movement]
    }
    for visual_movement in visual_movement_list:
        root_path = Path(
            f"{config['runs_root']}/{config['output_dir']}/{visual_movement}"
        )

        metrics_results = defaultdict(lambda: defaultdict(list))

        if visual_movement == "static":
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
                            instance_dir = category_dir / instance

                            if os.path.exists(instance_dir / "evaluation.json"):
                                try:
                                    with open(
                                        instance_dir / "evaluation.json",
                                        encoding="utf-8",
                                    ) as f:
                                        instance_result_dict = json.load(f)

                                    for (
                                        aspect,
                                        aspect_scores,
                                    ) in instance_result_dict.items():
                                        for (
                                            metric_name,
                                            metric_score,
                                        ) in aspect_scores.items():
                                            if not metric_score:
                                                continue

                                            metrics_results[aspect][metric_name].append(
                                                metric_score
                                            )
                                except Exception as e:
                                    pass
            calculate_mean_scores(
                metrics_results, visual_movement, scores, calculate_raw_score
            )

        elif visual_movement == "dynamic":
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
                        instance_dir = motion_type_dir / instance

                        if os.path.exists(instance_dir / "evaluation.json"):
                            try:
                                with open(
                                    instance_dir / "evaluation.json", encoding="utf-8"
                                ) as f:
                                    instance_result_dict = json.load(f)

                                for (
                                    aspect,
                                    aspect_scores,
                                ) in instance_result_dict.items():
                                    for (
                                        metric_name,
                                        metric_score,
                                    ) in aspect_scores.items():
                                        if not metric_score:
                                            continue

                                        metrics_results[aspect][metric_name].append(
                                            metric_score
                                        )
                            except Exception as e:
                                pass
            calculate_mean_scores(
                metrics_results, visual_movement, scores, calculate_raw_score
            )

    if not calculate_raw_score:
        metrics_static = worldscore_list["static"]
        metrics_dynamic = worldscore_list["dynamic"]
        worldscore_static = round(
            np.mean(np.array([scores[metric] for metric in metrics_static])).item(), 2
        )
        worldscore_dynamic = round(
            np.mean(
                np.array(
                    [scores[metric] for metric in metrics_static + metrics_dynamic]
                )
            ).item(),
            2,
        )
        print_banner("RESULT")
        print(model_name)
        print(f"Worldscore-Static: {worldscore_static}")
        print(f"Worldscore-Dynamic: {worldscore_dynamic}")
        print(f"Scores: {scores}")
        scores["WorldScore-Static"] = worldscore_static
        scores["WorldScore-Dynamic"] = worldscore_dynamic
        with open(
            os.path.join(config["runs_root"], config["output_dir"], "worldscore.json"),
            "w",
        ) as f:
            json.dump(scores, f, indent=4)
    else:
        print_banner("RESULT")
        print(model_name)
        print(f"Raw Scores: {scores}")
        with open(
            os.path.join(
                config["runs_root"], config["output_dir"], "worldscore_raw.json"
            ),
            "w",
        ) as f:
            json.dump(scores, f, indent=4)

    return


def run_evaluation(
    args: Namespace,
    num_jobs: int = 1000,
    use_slurm: bool = False,
    only_calculate_mean: bool = False,
    delete_calculated_results: bool = False,
    **slurm_parameters: dict,
) -> None:
    from worldscore.benchmark.helpers.evaluator import Evaluator

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
    evaluator.evaluate(
        num_jobs=num_jobs,
        use_slurm=use_slurm,
        only_calculate_mean=only_calculate_mean,
        delete_calculated_results=delete_calculated_results,
        **slurm_parameters,
    )


def main(
    model_name: str,
    num_jobs: int = 1000,
    use_slurm: bool = False,
    only_calculate_mean: bool = False,
    delete_calculated_results: bool = False,
    **slurm_parameters: dict,
) -> None:
    assert check_model(model_name), "Model not exists!"
    model_type = get_model2type(type2model)[model_name]
    if model_type == "threedgen":
        visual_movement_list = ["static"]
    else:
        visual_movement_list = ["static", "dynamic"]

    for visual_movement in visual_movement_list:
        args = Namespace(
            model_name=model_name,
            visual_movement=visual_movement,
        )

        print_banner("EVALUATION")

        # if argv is None:
        #     argv = sys.argv[1:]
        # if '--help' in argv or '-h' in argv:
        #     parser.print_help()
        #     return

        # args = parser.parse_args(argv)

        run_evaluation(
            args,
            num_jobs=num_jobs,
            use_slurm=use_slurm,
            only_calculate_mean=only_calculate_mean,
            delete_calculated_results=delete_calculated_results,
            **slurm_parameters,
        )

    # Calculate worldscore
    calculate_worldscore(model_name, visual_movement_list)


if __name__ == "__main__":
    fire.Fire(main)
