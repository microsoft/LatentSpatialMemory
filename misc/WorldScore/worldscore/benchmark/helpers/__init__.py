import os

from omegaconf import OmegaConf

from worldscore.benchmark.helpers.dataloaders import dataloader_general
from worldscore.benchmark.helpers.helper import Helper
from worldscore.benchmark.utils.get_utils import get_dataloader
from worldscore.benchmark.utils.utils import check_model


def GetHelpers(model_name, visual_movement, json_file=""):
    assert check_model(model_name), "Model not exists!"

    ### Get dataloader, helper for model runing
    if json_file == "":
        json_file = f"{visual_movement}.json"
    root_path = os.getenv("WORLDSCORE_PATH")
    dataset_root = os.getenv("DATA_PATH")
    json_path = os.path.join(
        dataset_root, "WorldScore-Dataset", visual_movement, json_file
    )

    base_config = OmegaConf.load(f"{root_path}/WorldScore/config/base_config.yaml")
    config = OmegaConf.load(
        os.path.join(
            f"{root_path}/WorldScore/config/model_configs", f"{model_name}.yaml"
        )
    )
    config = OmegaConf.merge(base_config, config)
    config.json_path = json_path
    config.visual_movement = visual_movement
    # Interpolate environment variables in the YAML file
    config = OmegaConf.to_container(config, resolve=True)

    loader = get_dataloader(config)
    helper = Helper(config)
    dataloader = loader.data

    return dataloader, helper


def GetDataloader(visual_movement, json_file=None, noise=False, noise_type="simple"):
    ### Get dataloader for data analysis
    if json_file is None:
        json_file = f"{visual_movement}.json"
    root_path = os.getenv("WORLDSCORE_PATH")
    dataset_root = os.getenv("DATA_PATH")
    json_path = os.path.join(
        dataset_root, "WorldScore-Dataset", visual_movement, json_file
    )

    config = OmegaConf.load(f"{root_path}/WorldScore/config/base_config.yaml")

    config.json_path = json_path
    config.visual_movement = visual_movement
    config.noise = noise
    config.noise_type = noise_type
    # Interpolate environment variables in the YAML file
    config = OmegaConf.to_container(config, resolve=True)

    loader = dataloader_general(config)

    return loader.data
