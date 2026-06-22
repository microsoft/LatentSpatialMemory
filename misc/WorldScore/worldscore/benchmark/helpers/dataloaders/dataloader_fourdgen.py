import json
import os

from tqdm import tqdm

from worldscore.benchmark.helpers.dataloader import DataLoader


class dataloader_fourdgen(DataLoader):
    def __init__(self, evaluation_config):
        super().__init__(evaluation_config)
        self.load(regenerate=evaluation_config["regenerate"])

    def load(self, regenerate=False):
        json_path = self.evaluation_config["json_path"]
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} not found.")

        with open(json_path, "rb") as f:
            data = json.load(f)

        for image_data in tqdm(data):
            data_prepared = self.load_data(image_data, regenerate)
            if image_data["visual_movement"] == "static":
                data_point, _, _, ignore = data_prepared
            else:
                data_point, ignore = data_prepared

            if ignore:
                continue

            self.data.append(data_point)
        return self.data
