import os

from worldscore.benchmark.helpers.camera_generator import CameraGen
from worldscore.benchmark.utils.utils import get_model2type, get_scene_num, type2model


def generate_prompt_list(prompt_list, style):
    new_prompt_list = []
    for prompt in prompt_list:
        if prompt.endswith("."):
            new_prompt = prompt + " " + style
        else:
            new_prompt = prompt + ". " + style
        new_prompt_list.append(new_prompt)
    return new_prompt_list


class DataLoader:
    def __init__(self, evaluation_config, verbose=False):
        self.evaluation_config = evaluation_config
        self.model_name = evaluation_config["model"]
        self.model_type = get_model2type(type2model)[self.model_name]
        self.visual_movement = None
        self.visual_style = None
        self.scene_type = None
        self.motion_type = None
        self.camera_path = None
        self.category = None
        self.name = None
        self.interpframe_num = None
        self.anchor_frame_idx = None
        self.scene_num = None
        self.data = []
        self.verbose = verbose

        self.cam_gen = CameraGen(evaluation_config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_config(self):
        return self.evaluation_config

    def set_interpframe_num(self):
        self.anchor_frame_idx = [0]

        self.scene_num = get_scene_num(self.camera_path)
        for _ in range(self.scene_num):
            self.anchor_frame_idx.append(
                self.anchor_frame_idx[-1] + self.evaluation_config["frames"] - 1
            )
        assert len(self.anchor_frame_idx) == self.scene_num + 1
        self.interpframe_num = (
            self.scene_num * (self.evaluation_config["frames"] - 1) + 1
        )
        return self.interpframe_num

    def data_exists(self, root_path):
        self.set_interpframe_num()
        if not os.path.exists(root_path):
            if self.verbose:
                print(f"-- Root path does not exist: {root_path}")
            return False

        dirs = os.listdir(root_path)

        if self.visual_movement == "static":
            dirs_should_be = [
                "frames",
                "image_data.json",
                "camera_data.json",
                "input_image.png",
            ]
        elif self.visual_movement == "dynamic":
            dirs_should_be = [
                "frames",
                "image_data.json",
                # "camera_data.json",
                "input_image.png",
            ]

        for dir_name in dirs_should_be:
            if dir_name not in dirs:
                if self.verbose:
                    print(f"-- Directory/file {dir_name} does not exist in {root_path}")
                return False
            if ".json" in dir_name or ".png" in dir_name:
                continue
            # now check the number of files in each directory
            file_list = os.listdir(os.path.join(root_path, dir_name))
            file_len = len(
                [f for f in file_list if f.endswith(".png") or f.endswith(".jpg")]
            )
            if dir_name == "frames":
                if file_len < self.interpframe_num:
                    if self.verbose:
                        print(f"-- Directory {dir_name} is incomplete in {root_path}")
                    return False
                continue

        return True

    def load_data(self, image_data, regenerate):
        ignore = False
        if image_data["visual_movement"] == "static":
            (
                self.visual_movement,
                self.visual_style,
                self.scene_type,
                self.camera_path,
            ) = (
                image_data["visual_movement"],
                image_data["visual_style"],
                image_data["scene_type"],
                image_data["camera_path"],
            )
            self.category = image_data["image"].split("/")[-2]
            self.name = image_data["image"].split("/")[-1].split(".")[0]

            runs_root = self.evaluation_config["runs_root"]
            output_dir = self.evaluation_config["output_dir"]
            root_path = os.path.join(
                runs_root,
                output_dir,
                f"{self.visual_movement}/{self.visual_style}/{self.scene_type}/{self.category}/{self.name}",
            )

            if not regenerate and self.data_exists(root_path):
                if self.verbose:
                    print(
                        f"-- Data already exists for {self.visual_movement}-{self.visual_style}-{self.scene_type}-{self.category}-{self.name}"
                    )
                ignore = True
                return (None, None, None, ignore)

            image_path = os.path.join(
                self.evaluation_config["dataset_root"], image_data["image"]
            )
            inpainting_prompt_list = generate_prompt_list(
                image_data["prompt_list"], image_data["style"]
            )

            cameras, cameras_interp = self.cam_gen.generate_cameras(
                self.camera_path, root_path, verbose=self.verbose
            )

            data_point = {
                "image_path": image_path,
                "inpainting_prompt_list": inpainting_prompt_list,
                "total_frames": self.interpframe_num,
                "output_dir": root_path,
                "num_scenes": self.scene_num,
                "image_data": image_data,
                "camera_path": self.camera_path,
                "anchor_frame_idx": self.anchor_frame_idx,
            }
            return (data_point, cameras, cameras_interp, ignore)
        elif image_data["visual_movement"] == "dynamic":
            (
                self.visual_movement,
                self.visual_style,
                self.motion_type,
                self.camera_path,
            ) = (
                image_data["visual_movement"],
                image_data["visual_style"],
                image_data["motion_type"],
                image_data["camera_path"],
            )
            self.name = image_data["image"].split("/")[-1].split(".")[0]

            runs_root = self.evaluation_config["runs_root"]
            output_dir = self.evaluation_config["output_dir"]
            root_path = os.path.join(
                runs_root,
                output_dir,
                f"{self.visual_movement}/{self.visual_style}/{self.motion_type}/{self.name}",
            )

            if not regenerate and self.data_exists(root_path):
                print(
                    f"-- Data already exists for {self.visual_movement}-{self.visual_style}-{self.motion_type}-{self.name}"
                )
                ignore = True
                return (None, ignore)

            if self.evaluation_config["model"] == "dreammachine_i2v":
                image_path = image_data.get("image_url", None)
                if not image_path:
                    ignore = True
                    return (None, ignore)
            else:
                image_path = os.path.join(
                    self.evaluation_config["dataset_root"], image_data["image"]
                )
            inpainting_prompt_list = generate_prompt_list(
                [image_data["prompt"]], image_data["style"]
            )

            data_point = {
                "image_path": image_path,
                "inpainting_prompt_list": inpainting_prompt_list,
                "total_frames": self.interpframe_num,
                "output_dir": root_path,
                "num_scenes": self.scene_num,
                "image_data": image_data,
                "camera_path": self.camera_path,
            }
            return (data_point, ignore)
