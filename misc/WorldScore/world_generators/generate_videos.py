# """Inference script for generate videos for the WorldScore benchmark."""

import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import fire
import hydra
import omegaconf
import structlog
import submitit

from worldscore.benchmark.helpers import GetHelpers
from worldscore.benchmark.utils.utils import check_model, get_model2type, type2model

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

logger = structlog.getLogger()


def generate_video(generator, model_helper, instance, generation_type):
    """Generates the video for a single prompt instance."""
    # Convert instance into the format expected by the model.
    if generation_type == "i2v":
        image_path, prompt_list = model_helper.adapt(instance)
    elif generation_type == "t2v":
        prompt_list = model_helper.adapt(instance)
        image_path = None
    else:
        raise ValueError()

    all_generated_frames = []
    for i, prompt in enumerate(prompt_list):
        generated_frames = generator.generate_video(
            prompt=prompt, image_path=image_path
        )

        # Update the prompt image with the last generated image in the path.
        if generation_type == "i2v":
            image_path = model_helper.save_image(
                generated_frames[-1], image_path, i + 1
            )

        # Append generated images to all_generated_images. Note that we use
        # the last frame of the generation for prompting, so it's the same as
        # the first frame of the next generation. As a result, we skip it.
        if i == 0:
            all_generated_frames += generated_frames
        else:
            all_generated_frames += generated_frames[1:]

    # benchmark output
    logger.info("Save generated frames", num_frames=len(all_generated_frames))
    model_helper.save(all_generated_frames)


def process_batch(
    model_helper: Any, data_batch: Sequence[Any], model_config: omegaconf.DictConfig
):
    """
    This is a very thin wrapper around the generated video function to
    allow us to easily run on SLURM with submitit for batches of prompts.
    """
    # Initialize the model once with its specific configs
    logger.info("Initialized model", model_name=model_config.model_name)
    generation_type = model_config.generation_type
    generator = hydra.utils.instantiate(model_config)

    # Iterate over batch -- this should be the same across many models.
    for instance in data_batch:
        try:
            generate_video(generator, model_helper, instance, generation_type)
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.error("Failed to process instance", exc_info=err)


def create_slurm_executor(
    log_dir: str,
    cpus_per_task: int = 8,
    gpus_per_node: int = 1,
    timeout_min: int = 600,
    mem_gb: int = 128,
    slurm_array_parallelism: int = 256,
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
        setup=["cd $SUBMITIT_LOCAL_JOB_DIR/../../.."],
        **slurm_parameters,
    )
    return submitit_executor


def main(
    model_name: str,
    prompt_set: str = "",
    num_jobs: int = 1000,
    use_slurm: bool = False,
    **slurm_parameters: dict,
):
    """
    Main function for generating videos using world score prompts and a specific model
    config

    Args:
        model_name: Model used for generation (has to match hydra config)
        prompt_set: JSON file for prompt set
        num_jobs: Number of jobs to be used for generation.
        use_slurm: Whether to run the generation on slurm. Defaults to False.
        slurm_parameters: keywork arguments for be passed to slurm.
    """

    assert check_model(model_name), "Model not exists!"
    model_type = get_model2type(type2model)[model_name]
    if model_type == "threedgen":
        visual_movement_list = ["static"]
    else:
        visual_movement_list = ["static", "dynamic"]

    for visual_movement in visual_movement_list:
        data_instances, helper = GetHelpers(model_name, visual_movement, prompt_set)

        # Load model config from YAML
        config_path = Path(__file__).parent / "configs" / f"{model_name}.yaml"
        model_config = omegaconf.OmegaConf.load(config_path)

        # Batch the instances for different workers.
        num_instances = len(data_instances)
        batch_size = max(len(data_instances) // num_jobs + 1, 5)
        data_batches = [
            data_instances[start_idx : start_idx + batch_size]
            for start_idx in range(0, num_instances, batch_size)
        ]
        logger.info(
            "Created data batches",
            num_instances=num_instances,
            num_jobs=len(data_batches),
            batch_size=batch_size,
        )

        # Generative videos
        if use_slurm:
            # Set log directory
            root_dir = Path(__file__).parent.parent
            curr_time = datetime.strftime(datetime.now(), "%Y_%d_%b_%H%M%S")
            log_dir = root_dir / f"submitit_logs/{curr_time}/"
            slurm_executor = create_slurm_executor(log_dir=log_dir, **slurm_parameters)
            logger.info("SLURM logging", log_dir=log_dir)

            # Launch on SLURM
            with slurm_executor.batch():
                jobs = [
                    slurm_executor.submit(
                        partial(
                            process_batch,
                            data_batch=data_batch,
                            model_config=model_config,
                            model_helper=helper,
                        )
                    )
                    for data_batch in data_batches
                ]

            submitit.helpers.monitor_jobs(jobs)
        else:
            _ = [
                process_batch(
                    data_batch=data_batch,
                    model_helper=helper,
                    model_config=model_config,
                )
                for data_batch in data_batches
            ]


if __name__ == "__main__":
    fire.Fire(main)
