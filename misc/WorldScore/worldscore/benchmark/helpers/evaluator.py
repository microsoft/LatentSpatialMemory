import json
import os
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import structlog
import submitit
import torch
from tqdm import tqdm

from worldscore.benchmark.metrics import (
    CameraErrorMetric,
    CLIPImageQualityAssessmentPlusMetric,
    CLIPScoreMetric,
    GramMatrixMetric,
    IQACLIPAestheticScoreMetric,
    MotionAccuracyMetric,
    MotionSmoothnessMetric,
    ObjectDetectionMetric,
    OpticalFlowAverageEndPointErrorMetric,
    OpticalFlowMetric,
    ReprojectionErrorMetric,
)
from worldscore.benchmark.utils.utils import (
    aspect_info,
    empty_cache,
    get_model2type,
    layout_info,
    type2model,
)
from worldscore.common.utils import print_banner

logger = structlog.getLogger()


def process_batch(
    config,
    instance_batch,
    aspect_list,
    visual_movement,
):
    evaluator = Evaluator(config)
    logger.info("Processing batch", model_name=evaluator.config["model"])
    for instance_attributes in instance_batch:
        if visual_movement == "static":
            visual_style, scene_type, category, instance, instance_dir = (
                instance_attributes
            )

            if (instance_dir / "evaluation.json").exists():
                try:
                    with open(instance_dir / "evaluation.json", encoding="utf-8") as f:
                        evaluation_data = json.load(f)
                    for aspect in evaluation_data.keys():
                        for metric_name in evaluation_data[aspect].keys():
                            evaluator.metrics_results[visual_style][scene_type][
                                category
                            ][instance][aspect][metric_name]["score"] = evaluation_data[
                                aspect
                            ][metric_name]["score"]
                            evaluator.metrics_results[visual_style][scene_type][
                                category
                            ][instance][aspect][metric_name][
                                "score_normalized"
                            ] = evaluation_data[aspect][metric_name]["score_normalized"]
                except Exception as e:
                    pass

            for aspect in aspect_list:
                metric_list = aspect_info[aspect]["metrics"]
                metric_type = aspect_info[aspect]["type"]
                for metric, metric_attribute in metric_list.items():
                    evaluator.aspect_evaluator = get_aspect_evaluator(
                        metric,
                        metric_type,
                        metric_attribute,
                        aspect,
                        evaluator.generate_type,
                    )

                    print("-- Aspect (metric):", f"{aspect} ({metric})")

                    with open(instance_dir / "camera_data.json", encoding="utf-8") as f:
                        camera_data = json.load(f)

                    with open(instance_dir / "image_data.json", encoding="utf-8") as f:
                        image_data = json.load(f)

                    camera_movement = image_data["camera_path"][0]
                    camera_movement_type = layout_info[camera_movement]["layout_type"]
                    if camera_movement_type == "intra" and aspect in [
                        "content_alignment",
                        "content_control",
                    ]:
                        print(
                            f"Skipping {aspect} for {camera_movement}({camera_movement_type}) layout"
                        )
                        continue

                    total_frames = image_data["total_frames"]
                    text_prompt_list = image_data["prompt_list"]
                    num_scenes = image_data["num_scenes"]
                    anchor_frame_idx = image_data["anchor_frame_idx"]
                    content_list = image_data["content_list"]

                    frames_dir = instance_dir / "frames"
                    # Sorting is crucial! Video frames must be evaluated in
                    # order.
                    image_list = sorted(
                        [
                            str(image_path.absolute())
                            for image_path in frames_dir.iterdir()
                            if image_path.suffix in [".jpg", ".png"]
                        ]
                    )

                    reference_image = instance_dir / "input_image.png"

                    score, score_normalized = evaluator.calculate_metrics(
                        camera_data=camera_data,
                        num_scenes=num_scenes,
                        total_frames=total_frames,
                        anchor_frame_idx=anchor_frame_idx,
                        text_prompt_list=text_prompt_list,
                        image_list=image_list,
                        reference_image=reference_image,
                        masked_images_all=None,
                        content_list=content_list,
                        objects_list=None,
                    )

                    evaluator.metrics_results[visual_style][scene_type][category][
                        instance
                    ][aspect][evaluator.aspect_evaluator.name]["score"] = score
                    evaluator.metrics_results[visual_style][scene_type][category][
                        instance
                    ][aspect][evaluator.aspect_evaluator.name][
                        "score_normalized"
                    ] = score_normalized

                    empty_cache()

            # save results
            instance_result_dict = evaluator.metrics_results[visual_style][scene_type][
                category
            ][instance]
            result_file_name = "evaluation.json"
            result_path = instance_dir / result_file_name
            with open(result_path, mode="w", encoding="utf-8") as f:
                json.dump(instance_result_dict, f, indent=4)

        elif visual_movement == "dynamic":
            visual_style, motion_type, instance, instance_dir = instance_attributes

            if (instance_dir / "evaluation.json").exists():
                try:
                    with open(instance_dir / "evaluation.json", encoding="utf-8") as f:
                        evaluation_data = json.load(f)
                    for aspect in evaluation_data.keys():
                        for metric_name in evaluation_data[aspect].keys():
                            evaluator.metrics_results[visual_style][motion_type][
                                instance
                            ][aspect][metric_name]["score"] = evaluation_data[aspect][
                                metric_name
                            ]["score"]
                            evaluator.metrics_results[visual_style][motion_type][
                                instance
                            ][aspect][metric_name][
                                "score_normalized"
                            ] = evaluation_data[aspect][metric_name]["score_normalized"]
                except Exception as e:
                    pass

            for aspect in aspect_list:
                metric_list = aspect_info[aspect]["metrics"]
                metric_type = aspect_info[aspect]["type"]
                for metric, metric_attribute in metric_list.items():
                    evaluator.aspect_evaluator = get_aspect_evaluator(
                        metric,
                        metric_type,
                        metric_attribute,
                        aspect,
                        evaluator.generate_type,
                    )

                    print("-- Aspect (metric):", f"{aspect} ({metric})")

                    with open(instance_dir / "image_data.json", encoding="utf-8") as f:
                        image_data = json.load(f)

                    total_frames = image_data["total_frames"]
                    text_prompt_list = [image_data["prompt"]]
                    num_scenes = image_data["num_scenes"]
                    masked_images_all = [
                        os.path.join(evaluator.dataset_root_path, masked_image)
                        for masked_image in image_data["masks"]
                    ]
                    objects_list = image_data["objects"]

                    frames_dir = instance_dir / "frames"
                    # Sorting is crucial! Video frames must be evaluated in
                    # order.
                    image_list = sorted(
                        [
                            str(image_path.absolute())
                            for image_path in frames_dir.iterdir()
                            if image_path.suffix in [".jpg", ".png"]
                        ]
                    )

                    reference_image = instance_dir / "input_image.png"

                    score, score_normalized = evaluator.calculate_metrics(
                        camera_data=None,
                        num_scenes=num_scenes,
                        total_frames=total_frames,
                        anchor_frame_idx=None,
                        text_prompt_list=text_prompt_list,
                        image_list=image_list,
                        reference_image=reference_image,
                        masked_images_all=masked_images_all,
                        content_list=None,
                        objects_list=objects_list,
                    )

                    evaluator.metrics_results[visual_style][motion_type][instance][
                        aspect
                    ][evaluator.aspect_evaluator.name]["score"] = score
                    evaluator.metrics_results[visual_style][motion_type][instance][
                        aspect
                    ][evaluator.aspect_evaluator.name][
                        "score_normalized"
                    ] = score_normalized

                    empty_cache()

            # save results
            instance_result_dict = evaluator.metrics_results[visual_style][motion_type][
                instance
            ]
            result_file_name = "evaluation.json"
            result_path = instance_dir / result_file_name
            with open(result_path, mode="w", encoding="utf-8") as f:
                json.dump(instance_result_dict, f, indent=4)

    return evaluator.metrics_results


def create_slurm_executor(
    log_dir: str,
    cpus_per_task: int = 8,
    gpus_per_node: int = 1,
    timeout_min: int = 300,
    mem_gb: int = 128,
    slurm_array_parallelism: int = 128,
    **slurm_parameters,
) -> submitit.AutoExecutor:
    """Create a Slurm executor with specified parameters.

    Args:
        log_dir: Directory where Slurm logs will be saved
        cpus_per_task: Number of CPUs per task. Defaults to 8.
        gpus_per_node: Number of GPUs per node. Defaults to 1.
        timeout_min: Job timeout in minutes. Defaults to 3"0.
        mem_gb: Memory in GB. Defaults to 128.
        slurm_array_parallelism: Maximum number of concurrent jobs. Defaults to 128.
        slurm_parameters: Additional slurm parameter.

    Returns:
        submitit.AutoExecutor: Configured Slurm executor
    """
    # Create Submitit executor
    submitit_executor = submitit.AutoExecutor(folder=log_dir, cluster="slurm")
    submitit_executor.update_parameters(
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=cpus_per_task,
        gpus_per_node=gpus_per_node,
        mem_gb=mem_gb,
        timeout_min=timeout_min,
        slurm_array_parallelism=slurm_array_parallelism,
        **slurm_parameters,
    )
    return submitit_executor


class Metric:
    def __init__(self, metric, name, metric_type, metric_attribute, aspect):
        self.metric = metric
        self.name = name
        self.type = metric_type
        self.metric_attribute = metric_attribute
        self.aspect = aspect

    def compute_scores(self, *args):
        return self.metric._compute_scores(*args)  # pylint: disable=protected-access


def renormalize_score(score_dict):
    for aspect, aspect_score in score_dict.items():
        if aspect not in aspect_info.keys():
            continue
        for metric, score_ in aspect_score.items():
            score = score_["score"]
            higher_is_better = aspect_info[aspect]["metrics"][metric][
                "higher_is_better"
            ]

            if metric in [
                "clip_score",
                "clip_iqa+",
                "clip_aesthetic",
                "motion_accuracy",
                "optical_flow",
            ]:
                avg = aspect_info[aspect]["metrics"][metric]["avg"]
                std = aspect_info[aspect]["metrics"][metric]["std"]
                z_max = aspect_info[aspect]["metrics"][metric]["z_max"]
                z_min = aspect_info[aspect]["metrics"][metric]["z_min"]
                range = aspect_info[aspect]["metrics"][metric]["range"]

                x_prime = ((score - avg) / std - z_min) / (z_max - z_min)
                if not higher_is_better:
                    x_prime = 1 - x_prime
                normalized_score = range[0] + (range[1] - range[0]) * x_prime
                normalized_score = max(0.0, min(1.0, normalized_score))
            else:
                empirical_max = aspect_info[aspect]["metrics"][metric]["empirical_max"]
                empirical_min = aspect_info[aspect]["metrics"][metric]["empirical_min"]

                if metric in ["camera_error", "motion_smoothness"]:
                    normalized_scores = []
                    for (
                        score_i,
                        empirical_max_i,
                        empirical_min_i,
                        higher_is_better_i,
                    ) in zip(score, empirical_max, empirical_min, higher_is_better):
                        score_i = max(empirical_min_i, min(empirical_max_i, score_i))
                        normalized_score = (score_i - empirical_min_i) / (
                            empirical_max_i - empirical_min_i
                        )
                        if not higher_is_better_i:
                            normalized_score = 1 - normalized_score
                        normalized_scores.append(normalized_score)
                    if metric == "motion_smoothness":
                        normalized_score = sum(normalized_scores) / len(
                            normalized_scores
                        )
                    else:
                        normalized_score = np.prod(normalized_scores) ** (
                            1 / len(normalized_scores)
                        )

                else:
                    score = max(empirical_min, min(empirical_max, score))
                    normalized_score = (score - empirical_min) / (
                        empirical_max - empirical_min
                    )
                    if not higher_is_better:
                        normalized_score = 1 - normalized_score

            score_dict[aspect][metric]["score_normalized"] = normalized_score

    return score_dict


def normalize_score(score, aspect_evaluator):
    name = aspect_evaluator.name
    if name in [
        "clip_score",
        "clip_iqa+",
        "clip_aesthetic",
        "motion_accuracy",
        "optical_flow",
    ]:
        avg = aspect_evaluator.metric_attribute["avg"]
        std = aspect_evaluator.metric_attribute["std"]
        z_max = aspect_evaluator.metric_attribute["z_max"]
        z_min = aspect_evaluator.metric_attribute["z_min"]
        range = aspect_evaluator.metric_attribute["range"]

        x_prime = ((score - avg) / std - z_min) / (z_max - z_min)
        if not aspect_evaluator.metric_attribute["higher_is_better"]:
            x_prime = 1 - x_prime
        normalized_score = range[0] + (range[1] - range[0]) * x_prime
        normalized_score = max(0.0, min(1.0, normalized_score))
    else:
        if name in ["camera_error", "motion_smoothness"]:
            normalized_scores = []
            for score_i, empirical_max, empirical_min, higher_is_better in zip(
                score,
                aspect_evaluator.metric_attribute["empirical_max"],
                aspect_evaluator.metric_attribute["empirical_min"],
                aspect_evaluator.metric_attribute["higher_is_better"],
            ):
                score_i = max(empirical_min, min(empirical_max, score_i))
                normalized_score = (score_i - empirical_min) / (
                    empirical_max - empirical_min
                )
                if not higher_is_better:
                    normalized_score = 1 - normalized_score
                normalized_scores.append(normalized_score)
            if name == "motion_smoothness":
                normalized_score = sum(normalized_scores) / len(normalized_scores)
            else:
                normalized_score = np.prod(normalized_scores) ** (
                    1 / len(normalized_scores)
                )
        else:
            empirical_max = aspect_evaluator.metric_attribute["empirical_max"]
            empirical_min = aspect_evaluator.metric_attribute["empirical_min"]

            score = max(empirical_min, min(empirical_max, score))
            normalized_score = (score - empirical_min) / (empirical_max - empirical_min)
            if not aspect_evaluator.metric_attribute["higher_is_better"]:
                normalized_score = 1 - normalized_score

    return normalized_score


def get_aspect_evaluator(
    metric_name, metric_type, metric_attribute, aspect, generate_type
):
    if metric_name == "clip_score":
        metric = Metric(
            CLIPScoreMetric(), "clip_score", metric_type, metric_attribute, aspect
        )
    elif metric_name == "clip_iqa+":
        metric = Metric(
            CLIPImageQualityAssessmentPlusMetric(),
            "clip_iqa+",
            metric_type,
            metric_attribute,
            aspect,
        )
    elif metric_name == "clip_aesthetic":
        metric = Metric(
            IQACLIPAestheticScoreMetric(),
            "clip_aesthetic",
            metric_type,
            metric_attribute,
            aspect,
        )
    elif metric_name == "gram_matrix":
        metric = Metric(
            GramMatrixMetric(), "gram_matrix", metric_type, metric_attribute, aspect
        )
    elif metric_name == "camera_error":
        metric = Metric(
            CameraErrorMetric(), "camera_error", metric_type, metric_attribute, aspect
        )
    elif metric_name == "optical_flow_aepe":
        metric = Metric(
            OpticalFlowAverageEndPointErrorMetric(),
            "optical_flow_aepe",
            metric_type,
            metric_attribute,
            aspect,
        )
    elif metric_name == "object_detection":
        metric = Metric(
            ObjectDetectionMetric(),
            "object_detection",
            metric_type,
            metric_attribute,
            aspect,
        )
    elif metric_name == "reprojection_error":
        metric = Metric(
            ReprojectionErrorMetric(),
            "reprojection_error",
            metric_type,
            metric_attribute,
            aspect,
        )
    elif metric_name == "optical_flow":
        metric = Metric(
            OpticalFlowMetric(), "optical_flow", metric_type, metric_attribute, aspect
        )
    elif metric_name == "motion_accuracy":
        metric = Metric(
            MotionAccuracyMetric(generate_type),
            "motion_accuracy",
            metric_type,
            metric_attribute,
            aspect,
        )
    elif metric_name == "motion_smoothness":
        metric = Metric(
            MotionSmoothnessMetric(),
            "motion_smoothness",
            metric_type,
            metric_attribute,
            aspect,
        )
    else:
        raise NotImplementedError(f"Metric {metric_name} is not implemented!")
    return metric


def get_rendered_images(rendered_images_all, i, anchor_frame_idx):
    """Get anchor rendered image representing the ith scene based on anchor frame indices.
    For example, with anchor_frame_idx [0, 48, 96, 144]:
    - i=0 returns frames [0]     (first quarter)
    - i=1 returns frames [48]    (around second anchor)
    - i=2 returns frames [96]   (around third anchor)
    - i=3 returns frames [144]  (last quarter)

    Args:
        rendered_images_all (list[Image.Image]): All rendered images
        i (int): Current scene index
        anchor_frame_idx (list[int]): List of anchor frame indices
    Returns:
        rendered_images (list[Image.Image]): Batch of images for current scene
    """
    rendered_images = [rendered_images_all[anchor_frame_idx[i]]]

    return rendered_images

    """Get batch of rendered images for the ith scene based on anchor frame indices.
    For example, with anchor_frame_idx [0, 48, 96, 144]:
    - i=0 returns frames [0:24]     (first quarter)
    - i=1 returns frames [24:72]    (around second anchor)
    - i=2 returns frames [72:120]   (around third anchor)
    - i=3 returns frames [120:144]  (last quarter)

    Args:
        rendered_images_all (list[Image.Image]): All rendered images
        i (int): Current scene index
        anchor_frame_idx (list[int]): List of anchor frame indices
    Returns:
        rendered_images (list[Image.Image]): Batch of images for current scene
    """
    is_first = i == 0
    is_last = i == len(anchor_frame_idx) - 1

    if is_first:
        # For first scene, take frames from start to midpoint between first two anchors
        interval = anchor_frame_idx[1] - anchor_frame_idx[0]
        mid_point = anchor_frame_idx[0] + interval // 2
        rendered_images = rendered_images_all[:mid_point]
    elif is_last:
        # For last scene, take frames from midpoint of last two anchors to end
        interval = anchor_frame_idx[-1] - anchor_frame_idx[-2]
        mid_point = (
            anchor_frame_idx[-2] + (interval + 1) // 2
        )  # Ceiling division for odd intervals
        rendered_images = rendered_images_all[mid_point:]
    else:
        # For middle scenes, take frames between midpoints
        prev_interval = anchor_frame_idx[i] - anchor_frame_idx[i - 1]
        next_interval = anchor_frame_idx[i + 1] - anchor_frame_idx[i]

        #  Ceiling division for odd intervals
        start_mid = anchor_frame_idx[i - 1] + (prev_interval + 1) // 2
        end_mid = anchor_frame_idx[i] + next_interval // 2
        rendered_images = rendered_images_all[start_mid:end_mid]

    return rendered_images


def check_evaluation_completeness(aspect_list, instance_result_dict):
    instance_aspect_list = list(instance_result_dict.keys())
    for aspect in aspect_list:
        if aspect not in instance_aspect_list:
            return False
    return True


class BaseEvaluator:
    """Base Evaluator Class."""

    def __init__(self, config):
        self.config = config
        self.aspect_evaluator = None

        self.generate_type = config["generate_type"]


class Evaluator(BaseEvaluator):
    """Evaluator Class."""

    def __init__(self, config):
        super().__init__(config)
        self.root_path = Path(
            f"{config['runs_root']}/{config['output_dir']}/{config['visual_movement']}"
        )
        self.dataset_root_path = config["dataset_root"]
        self.visual_movement = config["visual_movement"]
        self.num_frames = config["frames"]
        self.model_type = get_model2type(type2model)[config["model"]]

        if self.visual_movement == "static":
            self.metrics_results = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: defaultdict(
                                lambda: defaultdict(lambda: defaultdict())
                            )
                        )
                    )
                )
            )
        elif self.visual_movement == "dynamic":
            self.metrics_results = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))
                    )
                )
            )

    def build_full_aspect_list(self):
        aspect_list = {
            "static": [
                # Control
                "camera_control",
                "object_control",
                "content_alignment",
                # Quality
                "3d_consistency",
                "photometric_consistency",
                "style_consistency",
                "subjective_quality",
            ],
            # Dynamics
            "dynamic": [
                "motion_accuracy",
                "motion_magnitude",
                "motion_smoothness",
            ],
        }
        aspect_list = aspect_list[self.visual_movement]
        return aspect_list

    def _calculate_mean_scores(self):
        """
        Computes the mean scores over subsets of the dataset categories.

        Returns:
            tuple:
                domain_scores: aggregates scores for each style/scene-type/metric.
                scores: aggregates score for each metric (over all instances).
        """

        def calculate_mean_scores(metric_scores):
            scores = []
            scores_normalized = []
            for metric_score in metric_scores:
                scores.append(metric_score["score"])
                scores_normalized.append(metric_score["score_normalized"])

            try:
                score = np.mean(np.array(scores), axis=0).item()
            except:
                score = np.mean(np.array(scores), axis=0).tolist()
            try:
                score_normalized = np.mean(np.array(scores_normalized), axis=0).item()
            except:
                score_normalized = np.mean(np.array(scores_normalized), axis=0).tolist()

            return {"score": score, "score_normalized": score_normalized}

        domain_scores = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )

        # Iterate through all layers of metrics_results
        if self.visual_movement == "static":
            for visual_style in self.metrics_results:
                for scene_type, scene_scores in self.metrics_results[
                    visual_style
                ].items():
                    for category_scores in scene_scores.values():
                        for instance_scores in category_scores.values():
                            for aspect, aspect_scores in instance_scores.items():
                                for metric_name, metric_score in aspect_scores.items():
                                    # Calculate mean for each metric's scores
                                    if (
                                        not metric_score
                                    ):  # Check if there are any scores
                                        continue

                                    # Domain scores are aggregated over categories & instances.
                                    domain_scores[visual_style][scene_type][aspect][
                                        metric_name
                                    ].append(metric_score)

        elif self.visual_movement == "dynamic":
            for visual_style in self.metrics_results:
                for motion_type, motion_scores in self.metrics_results[
                    visual_style
                ].items():
                    for instance_scores in motion_scores.values():
                        for aspect, aspect_scores in instance_scores.items():
                            for metric_name, metric_score in aspect_scores.items():
                                # Calculate mean for each metric's scores
                                if not metric_score:  # Check if there are any scores
                                    continue

                                # Domain scores are aggregated over categories & instances.
                                domain_scores[visual_style][motion_type][aspect][
                                    metric_name
                                ].append(metric_score)

        # calculate mean scores for each domain
        for visual_style in domain_scores:
            for scene_type, scene_scores in domain_scores[visual_style].items():
                for aspect, aspect_scores in scene_scores.items():
                    for metric_name, metric_scores in aspect_scores.items():
                        domain_scores[visual_style][scene_type][aspect][metric_name] = (
                            calculate_mean_scores(metric_scores)
                        )

        scores = defaultdict(lambda: defaultdict(list))
        for visual_style in domain_scores:
            for scene_scores in domain_scores[visual_style].values():
                for aspect, aspect_scores in scene_scores.items():
                    for metric_name, metric_score in aspect_scores.items():
                        scores[aspect][metric_name].append(metric_score)

        for aspect in scores:
            for metric_name, metric_scores in scores[aspect].items():
                scores[aspect][metric_name] = calculate_mean_scores(metric_scores)

        return domain_scores, scores

    def calculate_metrics(
        self,
        camera_data,
        num_scenes,
        total_frames,
        anchor_frame_idx,
        text_prompt_list,
        image_list,
        reference_image,
        masked_images_all=None,
        content_list=None,
        objects_list=None,
    ):
        def get_scores():
            if aspect_evaluator.type == "prompt":
                scores = []
                for i in tqdm(range(0, num_scenes + 1)):
                    rendered_images = get_rendered_images(
                        rendered_images_all, i, anchor_frame_idx
                    )
                    if aspect_evaluator.name == "object_detection":
                        score = aspect_evaluator.compute_scores(
                            rendered_images, content_list[i]
                        )
                    else:
                        score = aspect_evaluator.compute_scores(
                            rendered_images, text_prompt_list[i]
                        )
                    scores.append(score)
                score = sum(scores) / len(scores)
            elif aspect_evaluator.type == "base":
                if aspect_evaluator.name == "gram_matrix":
                    selected_images = [
                        rendered_images_all[i] for i in anchor_frame_idx[1:]
                    ]
                    score = aspect_evaluator.compute_scores(
                        reference_image, selected_images
                    )
                elif (
                    aspect_evaluator.name == "clip_iqa+"
                    or aspect_evaluator.name == "clip_aesthetic"
                ):
                    selected_images = [rendered_images_all[i] for i in anchor_frame_idx]
                    score = aspect_evaluator.compute_scores(selected_images)
                elif aspect_evaluator.name == "motion_accuracy":
                    score = aspect_evaluator.compute_scores(
                        rendered_images_all, masked_images_all, objects_list
                    )
                else:
                    score = aspect_evaluator.compute_scores(rendered_images_all)
            elif aspect_evaluator.type == "camera":
                score = aspect_evaluator.compute_scores(
                    rendered_images_all, cameras_gt, camera_scale
                )
            else:
                raise NotImplementedError(
                    f"Aspect type {aspect_evaluator.type} is not implemented!"
                )
            return score

        rendered_images_all = image_list
        if camera_data is not None:
            cameras_gt = camera_data["cameras_interp"]
            cameras_gt = torch.tensor(cameras_gt)  # [frames, 4, 4]
            camera_scale = camera_data.get("scale", 1.0)

            if len(cameras_gt) != total_frames or total_frames != len(
                rendered_images_all
            ):
                raise ValueError("Number of frames and cameras do not match.")

        aspect_evaluator = self.aspect_evaluator

        try:
            score = get_scores()
        except Exception as e:
            print(f"Error in {aspect_evaluator.name}: {e}")
            score = (
                -100 if aspect_evaluator.metric_attribute["higher_is_better"] else 100
            )

        score_normalized = normalize_score(score, aspect_evaluator)

        return score, score_normalized

    def data_exists(self, root_path):
        dirs = os.listdir(root_path)
        image_data_path = os.path.join(root_path, "image_data.json")
        if not os.path.exists(image_data_path):
            return False

        try:
            with open(image_data_path, encoding="utf-8") as f:
                image_data = json.load(f)
        except Exception as e:
            print(f"Error in {root_path}: {e}")
            return False

        total_frames = image_data["total_frames"]

        if self.visual_movement == "static":
            dirs_should_be = [
                "frames",
                "image_data.json",
                "camera_data.json",
                "input_image.png",
            ]
        elif self.visual_movement == "dynamic":
            dirs_should_be = ["frames", "image_data.json", "input_image.png"]

        for dir_name in dirs_should_be:
            if dir_name not in dirs:
                print(f"-- Directory/file {dir_name} does not exist in {root_path}")
                return False
            if ".json" in dir_name or ".png" in dir_name:
                continue
            # now check the number of files in each directory
            file_list = os.listdir(os.path.join(root_path, dir_name))
            file_len = len(
                [f for f in file_list if f.endswith(".png") or f.endswith(".jpg")]
            )
            if dir_name == "frames":
                if file_len < total_frames:
                    print(f"-- Directory {dir_name} is incomplete in {root_path}")
                    return False
                continue

        return True

    def save_results(self, domain_scores, scores):
        """Save the results for each instance across all metrics in a json file."""

        # calculate mean scores for each domain
        for visual_style in domain_scores:
            for scene_type, scene_scores in domain_scores[visual_style].items():
                result_file_name = f"evaluation_{visual_style}_{scene_type}.json"
                result_path = (
                    self.root_path / visual_style / scene_type / result_file_name
                )
                with open(result_path, mode="w", encoding="utf-8") as f:
                    json.dump(scene_scores, f, indent=4)

        result_file_name = f"evaluation_mean.json"
        result_path = self.root_path / result_file_name

        with open(result_path, mode="w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4)

    def _evaluate(
        self,
        reevaluate=False,
        num_jobs: int = 1000,
        use_slurm: bool = False,
        only_calculate_mean: bool = False,
        delete_calculated_results: bool = False,
        **slurm_parameters: dict,
    ):
        print(f"Evaluating {self.visual_movement} data...")
        aspect_list = self.build_full_aspect_list()
        instance_attributes = []

        if self.visual_movement == "static":
            visual_styles = sorted(
                [x.name for x in self.root_path.iterdir() if x.is_dir()]
            )

            for visual_style in visual_styles:
                visual_style_dir = self.root_path / visual_style
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

                            # check if the data is complete
                            if not self.data_exists(instance_dir):
                                continue

                            if (
                                os.path.exists(instance_dir / "evaluation.json")
                                and not reevaluate
                            ):
                                print(
                                    f"-- Evaluation already exists for {instance_dir}"
                                )
                                if delete_calculated_results:
                                    os.remove(instance_dir / "evaluation.json")
                                    continue
                                else:
                                    try:
                                        with open(
                                            instance_dir / "evaluation.json",
                                            encoding="utf-8",
                                        ) as f:
                                            instance_result_dict = json.load(f)
                                        if only_calculate_mean:
                                            instance_result_dict = renormalize_score(
                                                instance_result_dict
                                            )
                                            with open(
                                                instance_dir / "evaluation.json",
                                                mode="w",
                                                encoding="utf-8",
                                            ) as f:
                                                json.dump(
                                                    instance_result_dict, f, indent=4
                                                )
                                            print(
                                                f"-- Renormalized score for {instance_dir}"
                                            )
                                        self.metrics_results[visual_style][scene_type][
                                            category
                                        ][instance] = instance_result_dict
                                        continue
                                    except Exception as e:
                                        pass

                            instance_attributes.append(
                                [
                                    visual_style,
                                    scene_type,
                                    category,
                                    instance,
                                    instance_dir,
                                ]
                            )

        elif self.visual_movement == "dynamic":
            visual_styles = sorted(
                [x.name for x in self.root_path.iterdir() if x.is_dir()]
            )

            for visual_style in visual_styles:
                visual_style_dir = self.root_path / visual_style
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

                        # check if the data is complete
                        if not self.data_exists(instance_dir):
                            continue

                        if (
                            os.path.exists(instance_dir / "evaluation.json")
                            and not reevaluate
                        ):
                            print(f"-- Evaluation already exists for {instance_dir}")
                            if delete_calculated_results:
                                os.remove(instance_dir / "evaluation.json")
                                continue
                            else:
                                try:
                                    with open(
                                        instance_dir / "evaluation.json",
                                        encoding="utf-8",
                                    ) as f:
                                        instance_result_dict = json.load(f)
                                    if only_calculate_mean:
                                        instance_result_dict = renormalize_score(
                                            instance_result_dict
                                        )
                                        with open(
                                            instance_dir / "evaluation.json",
                                            mode="w",
                                            encoding="utf-8",
                                        ) as f:
                                            json.dump(instance_result_dict, f, indent=4)
                                        print(
                                            f"-- Renormalized score for {instance_dir}"
                                        )
                                    self.metrics_results[visual_style][motion_type][
                                        instance
                                    ] = instance_result_dict
                                    continue
                                except Exception as e:
                                    pass

                        instance_attributes.append(
                            [visual_style, motion_type, instance, instance_dir]
                        )

        num_instances = len(instance_attributes)
        batch_size = max(len(instance_attributes) // num_jobs + 1, 10)
        instance_batches = [
            instance_attributes[start_idx : start_idx + batch_size]
            for start_idx in range(0, num_instances, batch_size)
        ]
        logger.info(
            "Created data batches",
            num_instances=num_instances,
            num_jobs=len(instance_batches),
            batch_size=batch_size,
        )

        if (
            instance_batches
            and not only_calculate_mean
            and not delete_calculated_results
        ):
            # Start evaluation
            if use_slurm:
                # Set log directory
                root_dir = Path(os.getcwd())
                curr_time = datetime.strftime(datetime.now(), "%Y_%d_%b_%H%M%S")
                log_dir = root_dir / f"submitit_logs/{curr_time}/"
                slurm_executor = create_slurm_executor(
                    log_dir=log_dir, **slurm_parameters
                )
                logger.info("SLURM logging", log_dir=log_dir)

                # Launch on SLURM
                with slurm_executor.batch():
                    jobs = [
                        slurm_executor.submit(
                            partial(
                                process_batch,
                                config=self.config,
                                instance_batch=instance_batch,
                                aspect_list=aspect_list,
                                visual_movement=self.visual_movement,
                            )
                        )
                        for instance_batch in instance_batches
                    ]
                submitit.helpers.monitor_jobs(jobs)

                batch_results = []
                for job in jobs:
                    try:
                        result = job.result()
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Job failed with error: {e}")
            else:
                batch_results = [
                    process_batch(
                        config=self.config,
                        instance_batch=instance_batch,
                        aspect_list=aspect_list,
                        visual_movement=self.visual_movement,
                    )
                    for instance_batch in instance_batches
                ]
            for result in batch_results:
                self.metrics_results.update(result)

    def print_results(self, domain_scores, scores):
        print_banner("EVALUATION RESULTS")
        print(f"\nEvaluation Results -- {self.visual_movement}:")
        print(
            "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
        )
        print("Domain Scores:")
        for visual_style in domain_scores:
            for scene_type, scene_scores in domain_scores[visual_style].items():
                print(f"Visual Style: {visual_style} | Scene Type: {scene_type}")
                for aspect, aspect_scores in scene_scores.items():
                    print(f"  Aspect: {aspect}")
                    for metric_name, metric_score in aspect_scores.items():
                        print(f"    Metric: {metric_name}")
                        print(f"      Score: {metric_score['score']}")
                        print(
                            f"      Normalized Score: {metric_score['score_normalized']}"
                        )
        print(
            "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
        )
        print(f"Overall Scores -- {self.visual_movement}:")
        for aspect, aspect_scores in scores.items():
            print(f"Aspect: {aspect}")
            for metric_name, metric_score in aspect_scores.items():
                print(f"  Metric: {metric_name}")
                print(f"    Score: {metric_score['score']}")
                print(f"    Normalized Score: {metric_score['score_normalized']}")

    def evaluate(
        self,
        verbose=True,
        reevaluate=False,
        num_jobs: int = 1000,
        use_slurm: bool = False,
        only_calculate_mean: bool = False,
        delete_calculated_results: bool = False,
        **slurm_parameters: dict,
    ):
        """
        Evaluates all the generations across all aspects as defined in the
        `build_full_aspect_list` function.

        Args:
            save: Save the computed metrics. Defaults to True.

        Returns:
            tuple:
                domain_scores (dict): scores for each style/scene-type/metric.
                scores (dict): scores aggreagted for each metric.
        """

        self._evaluate(
            reevaluate,
            num_jobs,
            use_slurm,
            only_calculate_mean,
            delete_calculated_results,
            **slurm_parameters,
        )

        # domain_scores, scores = self._calculate_mean_scores()

        # self.save_results(domain_scores, scores)

        # if verbose and not delete_calculated_results:
        #     self.print_results(domain_scores, scores)

        return


class Analysis(BaseEvaluator):
    """Analysis Class."""

    def __init__(self, config, dataloader):
        super().__init__(config)
        self.dataloader = dataloader
        self.score_dict = None

        self.noise = config["noise"]
        self.noise_type = config["noise_type"]

    def _get_score_dict(self, aspect_list):
        score_dict = {}
        for aspect in aspect_list:
            metrics = aspect_info[aspect]["metrics"]
            score_dict[aspect] = {}
            for metric in metrics:
                score_dict[aspect][metric] = []
        self.score_dict = score_dict
        return score_dict

    def build_full_aspect_list(self):
        aspect_list = {
            "simple": [
                # Control
                "content_alignment",
                # Quality
                "perceptual_quality",
                "aesthetic",
            ],
            "comparison": [
                # Quality
                "3d_consistency",
                "semantic_consistency",
                "style_consistency",
                # "photometric_consistency",
            ],
        }
        return aspect_list[self.noise_type]

    def _calculate_mean_scores(self):
        """
        Computes the mean scores over subsets of the dataset categories.

        Returns:
            tuple:
                domain_scores: aggregates scores for each style/scene-type/metric.
                scores: aggregates score for each metric (over all instances).
        """

        for aspect in self.score_dict:
            for metric_name, metric_scores in self.score_dict[aspect].items():
                mean_score = np.mean(metric_scores, axis=0).item()
                max_score = np.max(metric_scores, axis=0).item()
                min_score = np.min(metric_scores, axis=0).item()
                self.score_dict[aspect][metric_name] = {
                    "mean": mean_score,
                    "max": max_score,
                    "min": min_score,
                }

        return self.score_dict

    def calculate_metrics(
        self,
        text_prompt_list,
        image_list,
    ):
        rendered_images_all = image_list

        aspect_evaluator = self.aspect_evaluator
        if aspect_evaluator.type == "prompt":
            score = aspect_evaluator.compute_scores(
                rendered_images_all, text_prompt_list[0]
            )
        elif aspect_evaluator.type == "base":
            if aspect_evaluator.name == "gram_matrix":
                score = aspect_evaluator.compute_scores(
                    rendered_images_all[0], [rendered_images_all[1]]
                )
            else:
                score = aspect_evaluator.compute_scores(rendered_images_all)
        else:
            raise NotImplementedError(
                f"Aspect type {aspect_evaluator.type} is not implemented for analysis!"
            )

        self.score_dict[aspect_evaluator.aspect][aspect_evaluator.name].append(score)
        return self.score_dict

    def analyze_static(self):
        self._get_score_dict(self.build_full_aspect_list())
        for aspect in self.score_dict:
            metric_list = aspect_info[aspect]["metrics"]
            metric_type = aspect_info[aspect]["type"]
            for metric, metric_attribute in metric_list.items():
                self.aspect_evaluator = get_aspect_evaluator(
                    metric, metric_type, metric_attribute, aspect, self.generate_type
                )

                for data_point in self.dataloader:
                    print("-- Aspect (metric):", f"{aspect} ({metric})")

                    text_prompt_list = data_point["inpainting_prompt_list"]
                    image_list = data_point["image_path"]
                    if isinstance(image_list, str):
                        image_list = [image_list]

                    self.calculate_metrics(
                        text_prompt_list,
                        image_list,
                    )
                    empty_cache()

    def print_results(self, scores):
        print_banner("ANALYSIS RESULTS")
        print(f"\nAnalysis Results:")
        print(
            "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
        )
        print(f"Overall Scores:")
        for aspect, aspect_scores in scores.items():
            print(f"Aspect: {aspect}")
            for metric_name, metric_score in aspect_scores.items():
                print(f"  Metric: {metric_name}")
                print(f"    Mean Score: {metric_score['mean']}")
                print(f"    Max Score: {metric_score['max']}")
                print(f"    Min Score: {metric_score['min']}")

    def analyze(self, verbose=True):
        """
        Analyzes all the raw dataset across some aspects as defined in the
        `build_full_aspect_list` function.

        Args:
            save: Save the computed metrics. Defaults to True.

        Returns:
            tuple:
                domain_scores (dict): scores for each style/scene-type/metric.
                scores (dict): scores aggreagted for each metric.
        """
        self.analyze_static()

        scores = self._calculate_mean_scores()

        if verbose:
            self.print_results(scores)

        return scores
