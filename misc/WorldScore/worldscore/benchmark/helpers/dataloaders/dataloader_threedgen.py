import json

from tqdm import tqdm

from worldscore.benchmark.helpers.dataloader import DataLoader
from worldscore.benchmark.utils.utils import create_cameras


class dataloader_threedgen(DataLoader):
    def __init__(self, evaluation_config):
        super().__init__(evaluation_config)
        self.load(regenerate=evaluation_config["regenerate"])

    def load(self, regenerate=False):
        with open(self.evaluation_config["json_path"], "rb") as f:
            data = json.load(f)

        for image_data in tqdm(data):
            data_point, cameras, cameras_interp, ignore = self.load_data(
                image_data, regenerate
            )
            if ignore:
                continue

            cameras = create_cameras(cameras)
            cameras_interp = create_cameras(cameras_interp)

            data_point["cameras"] = cameras
            data_point["cameras_interp"] = cameras_interp
            self.data.append(data_point)
        return self.data
