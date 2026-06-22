import gc
import json
import os

import torch
from PIL import Image

from worldscore.benchmark.utils.modeltype import type2model

layout_info = {
    "push_in": {"scenenum": 1, "prompt": "push in", "layout_type": "intra"},
    "pull_out": {"scenenum": 1, "prompt": "pull out", "layout_type": "inter"},
    "move_left": {"scenenum": 1, "prompt": "move left", "layout_type": "inter"},
    "move_right": {"scenenum": 1, "prompt": "move right", "layout_type": "inter"},
    "orbit_left": {"scenenum": 1, "prompt": "orbit left", "layout_type": "intra"},
    "orbit_right": {"scenenum": 1, "prompt": "orbit right", "layout_type": "intra"},
    "pan_left": {"scenenum": 1, "prompt": "pan left", "layout_type": "inter"},
    "pan_right": {"scenenum": 1, "prompt": "pan right", "layout_type": "inter"},
    "pull_left": {
        "scenenum": 1,
        "prompt": "move left, pull out, then pan left",
        "layout_type": "inter",
    },
    "pull_right": {
        "scenenum": 1,
        "prompt": "move right, pull out, then pan right",
        "layout_type": "inter",
    },
    "fixed": {"scenenum": 1, "prompt": "fixed", "layout_type": "camera fix"},
}


aspect_info = {
    "camera_control": {
        "type": "camera",
        "metrics": {
            # rotation error, translation error
            "camera_error": {
                "empirical_max": [15, 0.5],
                "empirical_min": [0, 0],
                "higher_is_better": [False, False],
            }
        },
    },
    "object_control": {
        "type": "prompt",
        "metrics": {
            "object_detection": {
                "empirical_max": 1,
                "empirical_min": 0,
                "higher_is_better": True,
            }
        },
    },
    "content_alignment": {
        "type": "prompt",
        "metrics": {
            "clip_score": {
                "avg": 26.67,
                "std": 0.8875,
                "z_max": 1.2741,
                "z_min": -1.5950,
                "range": [0.25, 0.75],
                "higher_is_better": True,
            }
        },
    },
    "3d_consistency": {
        "type": "base",
        "metrics": {
            "reprojection_error": {
                "empirical_max": 1.0719,
                "empirical_min": 0,
                "higher_is_better": False,
            }
        },
    },
    "photometric_consistency": {
        "type": "base",
        "metrics": {
            "optical_flow_aepe": {
                "empirical_max": 1.1920,
                "empirical_min": 0,
                "higher_is_better": False,
            }
        },
    },
    "style_consistency": {
        "type": "base",
        "metrics": {
            "gram_matrix": {
                "empirical_max": 0.0070,
                "empirical_min": 0,
                "higher_is_better": False,
            }
        },
    },
    "subjective_quality": {
        "type": "base",
        "metrics": {
            "clip_iqa+": {
                "avg": 0.5842,
                "std": 0.0441,
                "z_max": 1.8703,
                "z_min": -1.8342,
                "range": [0.25, 0.75],
                "higher_is_better": True,
            },
            "clip_aesthetic": {
                "avg": 5.5952,
                "std": 0.2561,
                "z_max": 1.9741,
                "z_min": -2.9342,
                "range": [0.25, 0.75],
                "higher_is_better": True,
            },
        },
    },
    "motion_accuracy": {
        "type": "base",
        "metrics": {
            "motion_accuracy": {
                "avg": -0.1965,
                "std": 2.3687,
                "z_max": 1.5180,
                "z_min": -1.6147,
                "range": [0.25, 0.75],
                "higher_is_better": True,
            }
        },
    },
    "motion_magnitude": {
        "type": "base",
        "metrics": {
            "optical_flow": {
                "avg": 3.2425,
                "std": 3.4505,
                "z_max": 2.7498,
                "z_min": -0.8638,
                "range": [0.25, 0.75],
                "higher_is_better": True,
            }
        },
    },
    "motion_smoothness": {
        "type": "base",
        "metrics": {
            "motion_smoothness": {
                "empirical_max": [82.4014, 1, 0.0228],
                "empirical_min": [0, 0.9224, 0],
                # MSE, SSIM, LPIPS
                "higher_is_better": [False, True, False],
            }
        },
    },
}


def framedict2frames(framedict):
    return [item for frames in framedict.values() for item in frames[:-1]]


def get_model2type(type2model):
    model2type = {}
    for model_type, models in type2model.items():
        for model in models:
            model2type[model] = model_type
    return model2type


def save_frames(frames, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(f"{save_dir}/{i:03d}.png")


def save_frames_dict(frames_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, frames in frames_dict.items():
        os.makedirs(f"{save_dir}/{i:03d}", exist_ok=True)
        for j, frame in enumerate(frames):
            frame.save(f"{save_dir}/{i:03d}/{j:03d}.png")


def save_cameras(cameras_interp, focal_length=500, save_dir=None, camera_type=None):
    os.makedirs(save_dir, exist_ok=True)
    data = {"focal_length": focal_length}

    matrices = []
    for camera in cameras_interp:
        try:
            extrinsics = save_camera_transform[camera_type](camera)
        except:
            extrinsics = camera
        matrices.append(extrinsics)
    data["cameras_interp"] = torch.cat(matrices, dim=0).cpu().numpy().tolist()
    print("-- interp cameras saved! length: ", len(matrices))

    output_path = f"{save_dir}/camera_data.json"
    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def save_video(video, path, fps=10, save_gif=False):
    success = False
    try:
        from torchvision.io import write_video
        from torchvision.transforms import ToTensor

        frames_tensor = []
        for frame in video:
            frame_tensor = ToTensor()(frame).unsqueeze(0)
            frames_tensor.append(frame_tensor)

        video_tensor = (
            (255 * torch.cat(frames_tensor, dim=0)).to(torch.uint8).detach().cpu()
        )

        video_tensor = video_tensor.permute(0, 2, 3, 1)

        codec_priority = ["libx264", "libopenh264"]
        video_options = {
            "crf": "23",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
            "preset": "slow",
        }
        for codec in codec_priority:
            try:
                write_video(
                    str(path),
                    video_tensor,
                    fps=fps,
                    video_codec=codec,
                    options=video_options,
                )
                success = True
                break
            except:
                continue
    except Exception as e:
        print("Failed to save video with torchvision", e)

    if not success:
        try:
            import subprocess
            from pathlib import Path

            temp_dir = Path(path).parent / "temp_frames"
            temp_dir.mkdir(exist_ok=True)

            for i, frame in enumerate(video):
                frame.save(str(temp_dir / f"frame_{i:04d}.png"))

            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(temp_dir / "frame_%04d.png"),
                "-q:v",
                "1",
                "-b:v",
                "8M",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "scale=in_w:in_h",
                str(path),
            ]

            subprocess.run(cmd, check=True)
            print(f"Successfully saved video to {path}")
        except Exception as e:
            print(f"Error with ffmpeg: {e}")
        finally:
            import shutil

            shutil.rmtree(temp_dir)

    if save_gif:
        gif_path = str(path).replace(".mp4", ".gif")
        video[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=video[1:],
            duration=1000 / fps,
            loop=0,
        )


def merge_video(frames, fps=10, save_dir=None, save_gif=False, save_reverse=False):
    os.makedirs(save_dir, exist_ok=True)

    video = frames
    video_reverse = frames[::-1]

    save_video(video, f"{save_dir}/output.mp4", fps=fps, save_gif=save_gif)
    if save_reverse:
        save_video(
            video_reverse, f"{save_dir}/output_reverse.mp4", fps=fps, save_gif=save_gif
        )


def convert_pytorch3d_blender(camera):
    transform_matrix_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
    transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)

    pt3d_to_blender = torch.diag(torch.tensor([-1.0, 1, -1, 1], device=camera.device))
    transform_matrix_w2c_blender = pt3d_to_blender @ transform_matrix_w2c_pt3d

    extrinsics = transform_matrix_w2c_blender
    # c2w
    c2w_extrinsics = torch.inverse(extrinsics).unsqueeze(0)
    return c2w_extrinsics


def convert_pytorch3d_tensor_blender(camera):
    transform_matrix_w2c_pt3d = camera.transpose(0, 1)

    pt3d_to_blender = torch.diag(torch.tensor([-1.0, 1, -1, 1], device=camera.device))
    transform_matrix_w2c_blender = pt3d_to_blender @ transform_matrix_w2c_pt3d

    extrinsics = transform_matrix_w2c_blender
    # c2w
    c2w_extrinsics = torch.inverse(extrinsics).unsqueeze(0)
    return c2w_extrinsics


def convert_tensor_blender(camera):
    transform_matrix_w2c_pt3d = camera

    pt3d_to_blender = torch.diag(torch.tensor([-1.0, 1, -1, 1], device=camera.device))
    transform_matrix_w2c_blender = pt3d_to_blender @ transform_matrix_w2c_pt3d

    extrinsics = transform_matrix_w2c_blender
    # c2w
    c2w_extrinsics = torch.inverse(extrinsics).unsqueeze(0)
    return c2w_extrinsics


def convert_blender_pytorch3d(camera):
    c2w = camera
    w2c_blender = torch.inverse(c2w)

    blender_to_pt3d = torch.diag(torch.tensor([-1.0, 1, -1, 1]))
    w2c_pt3d = blender_to_pt3d @ w2c_blender
    w2c_pt3d = w2c_pt3d.transpose(0, 1)

    return w2c_pt3d


def convert_blender_blender(camera):
    c2w = camera  # for save end, w2c
    w2c_pt3d = torch.inverse(c2w)  # for save end, c2w

    return w2c_pt3d  # for save end, c2w


def convert_blender_tensor(camera):
    c2w = camera
    w2c_blender = torch.inverse(c2w)

    blender_to_pt3d = torch.diag(torch.tensor([-1.0, 1, -1, 1]))
    w2c_pt3d = blender_to_pt3d @ w2c_blender

    return w2c_pt3d


save_camera_transform = {
    "pytorch3d": convert_pytorch3d_blender,
    "pytorch3d_tensor": convert_pytorch3d_tensor_blender,
    "blender": convert_blender_blender,
    "tensor": convert_tensor_blender,
}

camera_transform = {
    "pytorch3d": convert_blender_pytorch3d,
    "pytorch3d_tensor": convert_blender_pytorch3d,
    "blender": convert_blender_blender,
    "tensor": convert_blender_tensor,
}


def create_cameras(cameras_npy):
    cameras = []
    for cam in cameras_npy:
        cam = torch.from_numpy(cam).to(torch.float32)
        cameras.append(cam)

    return torch.stack(cameras)


def generate_prompt_from_list(prompt_list, camera_path, static=False):
    generation_prompt_list = []
    if static:
        for i, layout in enumerate(camera_path):
            prompt = ""
            layout_prompt = layout_info[layout]["prompt"]
            prompt = "Camera " + layout_prompt
            scene_prompt = prompt_list[i + 1]
            prompt += ". " + scene_prompt
            prompt += ". Everything in the scene remains completely motionless."
            generation_prompt_list.append(prompt)
    else:
        layout = camera_path[0]
        prompt = ""
        layout_prompt = layout_info[layout]["prompt"]
        prompt = "Camera " + layout_prompt
        scene_prompt = prompt_list[0]
        prompt += ". " + scene_prompt
        generation_prompt_list.append(prompt)

    print(generation_prompt_list)
    return generation_prompt_list


def get_scene_num(layout):
    scene_num = 0
    for layout_item in layout:
        scene_num += layout_info[layout_item]["scenenum"]
    return scene_num


def center_crop(image_path, resolution, output_dir):
    image = Image.open(image_path)
    width, height = image.size
    target_width, target_height = resolution

    # Calculate target aspect ratio
    target_ratio = target_width / target_height
    current_ratio = width / height

    # Calculate crop dimensions to match target ratio
    if current_ratio > target_ratio:
        # Image is too wide, crop width
        new_width = int(height * target_ratio)
        new_height = height
    else:
        # Image is too tall, crop height
        new_width = width
        new_height = int(width / target_ratio)

    # Calculate crop coordinates
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    # Perform center crop
    cropped_image = image.crop((left, top, right, bottom))
    cropped_image = cropped_image.resize(resolution)
    cropped_image.save(f"{output_dir}/input_image.png")
    return f"{output_dir}/input_image.png"


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def check_model(model_name: str):
    for model_list in type2model.values():
        if model_name in model_list:
            return True
    return False
