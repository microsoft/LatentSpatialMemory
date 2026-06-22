import json
import os
import time

import torch
from PIL import Image
from torchvision.transforms import ToPILImage

from worldscore.benchmark.utils.get_utils import get_adapter
from worldscore.benchmark.utils.utils import merge_video, save_frames


class Helper:
    def __init__(self, config):
        self.focal_length = config.get("focal_length", 500)
        self.path = None
        self.config = config
        self.adapter = get_adapter(config)
        self.data = None

        self.frames = config["frames"]
        self.num_scenes = None

        self.start_time = None
        self.end_time = None
        self.total_time = None

    def set_path(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.path = output_dir

    def store_data(self, data):
        visual_movement = data["image_data"]["visual_movement"]
        self.data = data["image_data"]
        self.data["num_scenes"] = data["num_scenes"]
        self.data["total_frames"] = data["total_frames"]
        if visual_movement == "static":
            self.data["anchor_frame_idx"] = data["anchor_frame_idx"]
        with open(f"{self.path}/image_data.json", "w") as f:
            json.dump(self.data, f, indent=4)

    def prepare_data(self, output_dir, data):
        self.set_path(output_dir)
        self.store_data(data)
        self.num_scenes = data["num_scenes"]

    def adapt(self, data):
        self.start_time = time.time()
        return self.adapter(self.config, data, self)

    def save_image(self, last_frame, image_path, i):
        # Get the directory and extension from the path
        directory = os.path.dirname(image_path)
        ext = os.path.splitext(image_path)[1]
        # Create new path with input_image_{i} format
        image_path = os.path.join(directory, f"input_image_{i}{ext}")

        if isinstance(last_frame, Image.Image):
            last_frame.save(image_path)
        elif isinstance(last_frame, torch.Tensor):  # [3, h, w] (0, 1)
            last_frame = ToPILImage()(last_frame)
            last_frame.save(image_path)
        return image_path

    def save(self, all_interpframes):
        self.end_time = time.time()
        with open(f"{self.path}/time.txt", "w") as f:
            f.write(f"generate_time: {self.end_time - self.start_time}\n")

        fps = self.config.get("fps", 10)
        frames = []
        for frame in all_interpframes:
            if isinstance(frame, Image.Image):
                pass
            elif isinstance(frame, torch.Tensor):  # [3, h, w] (0, 1)
                frame = ToPILImage()(frame)
            frames.append(frame)

        save_frames(frames, save_dir=f"{self.path}/frames")

        merge_video(frames, save_dir=f"{self.path}/videos", fps=fps)
