import torch

from worldscore.benchmark.utils.utils import camera_transform, center_crop


def cameras_transform(cameras_tensor, camera_type):
    cameras = []
    for cam in cameras_tensor:
        cam = camera_transform[camera_type](cam)
        cam = torch.inverse(cam)  # w2c
        R = cam[:3, :3].T.numpy()
        T = cam[3, :3].unsqueeze(-1).numpy()
        T[-1] = -T[-1]
        cameras.append([R, T])
    return cameras


def adapter_luciddreamer(config, data, helper):
    output_dir, image_path, inpainting_prompt_list, cameras, cameras_interp = (
        data["output_dir"],
        data["image_path"],
        data["inpainting_prompt_list"],
        data["cameras"],
        data["cameras_interp"],
    )

    camera_type = config["camera_type"]
    cameras = cameras_transform(cameras, camera_type)
    cameras_interp = cameras_transform(cameras_interp, camera_type)

    helper.prepare_data(output_dir, data)
    image_path = center_crop(image_path, config["resolution"], output_dir)
    return image_path, inpainting_prompt_list, cameras, cameras_interp
