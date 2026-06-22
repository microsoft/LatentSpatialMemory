import json
from pathlib import Path

import numpy as np

from worldscore.benchmark.helpers.camera_generator import CameraGen
from worldscore.benchmark.utils.utils import (
    center_crop,
    create_cameras,
    generate_prompt_from_list,
)


def _load_cameras_from_json(camera_data_path: Path):
    """Load saved camera trajectories from WorldScore's camera_data.json."""
    with open(camera_data_path, encoding="utf-8") as f:
        camera_data = json.load(f)

    cameras = create_cameras(np.array(camera_data["cameras"]))
    cameras_interp = create_cameras(np.array(camera_data["cameras_interp"]))
    return cameras, cameras_interp


def _get_or_create_cameras(config, data, output_dir: str):
    """
    Resolve camera tensors for one sample.

    Priority:
    1. Use cameras already attached by the dataloader.
    2. Use existing camera_data.json on disk.
    3. Generate cameras on the fly with CameraGen.
    """
    # Prefer already prepared camera tensors if the dataloader provides them.
    cameras = data.get("cameras")
    cameras_interp = data.get("cameras_interp")
    if cameras is not None and cameras_interp is not None:
        return cameras, cameras_interp

    camera_data_path = Path(output_dir) / "camera_data.json"
    if camera_data_path.exists():
        return _load_cameras_from_json(camera_data_path)

    # Dynamic videogen samples may not have camera_data.json pre-generated.
    cam_gen = CameraGen(config)
    cameras_np, cameras_interp_np = cam_gen.generate_cameras(
        data["camera_path"], output_dir, verbose=False
    )
    cameras = create_cameras(cameras_np)
    cameras_interp = create_cameras(cameras_interp_np)
    return cameras, cameras_interp


def adapter_spatia(config, data, helper):
    """
    Spatia-specific adapter for pose-conditioned videogen hard-coded runs.

    Returns:
    - conditioning image path (cropped),
    - prompt list for generation,
    - keyframe camera tensors,
    - interpolated camera tensors.
    """
    output_dir = data["output_dir"]
    image_path = data["image_path"]
    inpainting_prompt_list = data["inpainting_prompt_list"]
    camera_path = data["camera_path"]

    # Reuse WorldScore prompt construction logic for static/dynamic splits.
    inpainting_prompt = generate_prompt_from_list(
        inpainting_prompt_list,
        camera_path,
        static=True if config["visual_movement"] == "static" else False,
    )

    # Reuse helper API to write benchmark metadata and prepare output folder.
    helper.prepare_data(output_dir, data)
    # Reuse WorldScore's canonical input preprocessing.
    image_path = center_crop(image_path, config["resolution"], output_dir)
    # Ensure pose inputs are always available for the model.
    cameras, cameras_interp = _get_or_create_cameras(config, data, output_dir)
    return image_path, inpainting_prompt, cameras, cameras_interp
