import torch
from PIL import Image
from pytorch3d.renderer import PerspectiveCameras

from worldscore.benchmark.utils.utils import camera_transform, center_crop


def cameras_transform(cameras_tensor, init_focal_length, camera_type):
    cameras = []
    for cam in cameras_tensor:
        cam = camera_transform[camera_type](cam)

        K = torch.zeros((1, 4, 4), device="cuda")
        K[0, 0, 0] = init_focal_length
        K[0, 1, 1] = init_focal_length
        K[0, 0, 2] = 256
        K[0, 1, 2] = 256
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1
        R = cam[:3, :3].unsqueeze(0).to("cuda")
        T = cam[3, :3].unsqueeze(0).to("cuda")
        camera = PerspectiveCameras(
            K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device="cuda"
        )
        cameras.append(camera)
    return cameras


def adapter_wonderjourney(config, data, helper):
    output_dir, image_path, inpainting_prompt_list, cameras, cameras_interp = (
        data["output_dir"],
        data["image_path"],
        data["inpainting_prompt_list"],
        data["cameras"],
        data["cameras_interp"],
    )

    focal_length, camera_type = config["focal_length"], config["camera_type"]
    cameras = cameras_transform(cameras, focal_length, camera_type)
    cameras_interp = cameras_transform(cameras_interp, focal_length, camera_type)

    helper.prepare_data(output_dir, data)
    image_path = center_crop(image_path, config["resolution"], output_dir)
    start_keyframe = Image.open(image_path).convert("RGB").resize((512, 512))
    return start_keyframe, inpainting_prompt_list, cameras, cameras_interp
