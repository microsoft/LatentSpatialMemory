import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from worldscore.benchmark.helpers.dataloader import generate_prompt_list


class dataloader_general:
    def __init__(self, evaluation_config):
        self.evaluation_config = evaluation_config
        self.data = []
        self.root_path = evaluation_config["benchmark_root"]
        self.load()

    def add_noise(self, image_path, i):
        # create a random gaussian noise image with the same size as the original image
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = np.array(image)
        noise = np.random.normal(loc=128, scale=50, size=image.shape)
        noise = np.clip(noise, 0, 255)
        noise = noise.astype(np.uint8)

        new_image_path = os.path.join(
            self.root_path, "output", "noisy_images", f"{i:06d}.png"
        )
        Image.fromarray(noise).save(new_image_path)
        return new_image_path

    def load(self):
        with open(self.evaluation_config["json_path"], "rb") as f:
            data = json.load(f)

        if self.evaluation_config["noise"]:
            os.makedirs(
                os.path.join(self.root_path, "output", "noisy_images"), exist_ok=True
            )

        for i, image_data in enumerate(tqdm(data)):
            image_path = os.path.join(
                self.evaluation_config["dataset_root"], image_data["image"]
            )
            if self.evaluation_config["noise"]:
                old_image_path = image_path
                image_path = self.add_noise(image_path, i)
                if self.evaluation_config["noise_type"] == "comparison":
                    image_path = [old_image_path, image_path]
            if image_data["visual_movement"] == "static":
                inpainting_prompt_list = generate_prompt_list(
                    image_data["prompt_list"], image_data["style"]
                )
            elif image_data["visual_movement"] == "dynamic":
                inpainting_prompt_list = generate_prompt_list(
                    [image_data["prompt"]], image_data["style"]
                )

            data_point = {
                "image_path": image_path,
                "inpainting_prompt_list": inpainting_prompt_list,
            }

            self.data.append(data_point)
        return self.data
