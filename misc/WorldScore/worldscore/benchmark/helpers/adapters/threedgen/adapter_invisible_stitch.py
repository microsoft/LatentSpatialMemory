import torch
from pytorch3d.renderer import PerspectiveCameras

from worldscore.benchmark.utils.utils import camera_transform, center_crop


def cameras_transform(cameras_tensor, focal_length, camera_type):
    cameras = []
    for cam in cameras_tensor:
        cam = camera_transform[camera_type](cam)
        R = cam[:3, :3].unsqueeze(0).to("cuda")
        T = cam[3, :3].unsqueeze(0).to("cuda")
        camera = PerspectiveCameras(
            R=R,
            T=T,
            focal_length=torch.tensor([focal_length], dtype=torch.float32),
            principal_point=(((512 - 1) / 2, (512 - 1) / 2),),
            image_size=((512, 512),),
            device="cuda",
            in_ndc=False,
        )
        cameras.append(camera)
    return cameras


def adapter_invisible_stitch(config, data, helper):
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
    return image_path, inpainting_prompt_list, cameras, cameras_interp
