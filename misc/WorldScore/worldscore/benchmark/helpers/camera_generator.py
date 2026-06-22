import copy
import json
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from worldscore.benchmark.utils.utils import get_model2type, layout_info, type2model
from worldscore.common.gpu_utils import get_torch_device


def convert_extrinsics_blender(camera):
    transform_matrix_w2c_pt3d = camera.RT.to(torch.float32)

    pt3d_to_blender = torch.diag(
        torch.tensor([-1.0, 1.0, -1.0, 1.0], device=camera.device)
    )
    transform_matrix_w2c_blender = pt3d_to_blender @ transform_matrix_w2c_pt3d

    extrinsics = transform_matrix_w2c_blender
    # c2w
    c2w_extrinsics = torch.inverse(extrinsics).unsqueeze(0)
    return c2w_extrinsics


class CameraExtrinsics(object):
    def __init__(self, R=None, T=None):
        """
        - R (torch.Tensor): Rotation matrix of shape (3, 3)
        - T (torch.Tensor): Translation vector of shape (3, 1)
        """
        if R is None:
            R = torch.eye(3)
        self.R = R

        if T is None:
            T = torch.zeros((3, 1))
        self.T = T

        self.update_RT()
        self.device = self.RT.device

    def update_RT(self):
        RT = torch.cat((self.R, self.T), dim=1)
        last_row = torch.tensor(
            [[0, 0, 0, 1]], device=self.R.device, dtype=self.R.dtype
        )

        self.RT = torch.cat((RT, last_row), dim=0)
        return self.RT

    def update_R_T(self):
        self.R = self.RT[:3, :3]
        self.T = self.RT[:3, 3].unsqueeze(1)
        return self.R, self.T

    def set_R(self, R):
        self.R = R
        self.update_RT()

    def set_T(self, T):
        self.T = T
        self.update_RT()

    def set_RT(self, RT):
        self.RT = RT
        self.update_R_T()

    def to(self, device):
        self.R = self.R.to(device)
        self.T = self.T.to(device)
        self.RT = self.RT.to(device)
        self.device = device
        return self


class CameraGen(object):
    def __init__(self, config):
        super(CameraGen, self).__init__()
        self.device = get_torch_device()

        self.root_path = config["benchmark_root"]
        self.focal_length = config.get("focal_length", 500)
        self.camera_speed = config.get("camera_speed", 1)

        self.model = config["model"]

        self.model_type = get_model2type(type2model)[self.model]
        self.total_frames = config["frames"] - 1

        self.rotation_range_theta = 30

        current_camera = CameraExtrinsics().to(self.device)
        self.cameras = [current_camera]

        self.cameras_interp = []
        self.config = config

    def clear(self):
        current_camera = CameraExtrinsics().to(self.device)
        self.cameras = [current_camera]

        self.cameras_interp = []
        return

    def interp_cameras(self, kf1_camera, kf2_camera):
        R1 = kf1_camera.R.cpu().numpy()
        R2 = kf2_camera.R.cpu().numpy()
        T1 = kf1_camera.T.cpu().numpy()
        T2 = kf2_camera.T.cpu().numpy()

        quat1 = R.from_matrix(R1).as_quat()
        quat2 = R.from_matrix(R2).as_quat()

        cameras = []
        for i in range(1, self.total_frames):
            next_camera = CameraExtrinsics().to(self.device)
            t = i / self.total_frames

            slerp = Slerp([0, 1], R.from_quat([quat1, quat2]))
            R_interpolated = slerp([t]).as_matrix()[0]
            next_camera.set_R(torch.from_numpy(R_interpolated).to(self.device))

            T_interpolated = T1 + t * (T2 - T1)
            next_camera.set_T(torch.from_numpy(T_interpolated).to(self.device))
            cameras.append(copy.deepcopy(next_camera))

        return cameras

    def _move_left(self, camera=None, magnitude=1):
        if camera is None:
            camera = copy.deepcopy(self.cameras[-1])
        T = camera.T
        T[0] -= self.camera_speed * magnitude
        camera.set_T(T)
        return camera

    def _move_right(self, camera=None, magnitude=1):
        if camera is None:
            camera = copy.deepcopy(self.cameras[-1])
        T = camera.T
        T[0] += self.camera_speed * magnitude
        camera.set_T(T)
        return camera

    def _push_in(self, camera=None, magnitude=1):
        if camera is None:
            camera = copy.deepcopy(self.cameras[-1])
        T = camera.T
        T[2] -= self.camera_speed * magnitude
        camera.set_T(T)
        return camera

    def _pull_out(self, camera=None, magnitude=1):
        if camera is None:
            camera = copy.deepcopy(self.cameras[-1])
        T = camera.T
        T[2] += self.camera_speed * magnitude
        camera.set_T(T)
        return camera

    def _orbit_left(self, camera=None):
        if camera is None:
            camera = copy.deepcopy(self.cameras[-1])

        radius = self.camera_speed
        T_neg, T_pos, z = (
            copy.deepcopy(camera.T),
            copy.deepcopy(camera.T),
            copy.deepcopy(camera.T[2]),
        )
        T_neg[2] = radius - z
        T_pos[2] = z - radius
        cam_neg = CameraExtrinsics(R=camera.R, T=T_neg)
        cam_pos = CameraExtrinsics(R=camera.R, T=T_pos)

        theta = torch.deg2rad(torch.tensor(-self.rotation_range_theta))
        rotation_matrix = torch.tensor(
            [
                [torch.cos(theta), 0, -torch.sin(theta), 0],
                [0, 1, 0, 0],
                [torch.sin(theta), 0, torch.cos(theta), 0],
                [0, 0, 0, 1],
            ],
            device=self.device,
        )
        RT = (cam_neg.RT @ rotation_matrix) @ cam_pos.RT
        camera.set_RT(RT)
        return camera

    def _orbit_right(self, camera=None):
        if camera is None:
            camera = copy.deepcopy(self.cameras[-1])

        radius = self.camera_speed
        T_neg, T_pos, z = (
            copy.deepcopy(camera.T),
            copy.deepcopy(camera.T),
            copy.deepcopy(camera.T[2]),
        )
        T_neg[2] = radius - z
        T_pos[2] = z - radius
        cam_neg = CameraExtrinsics(R=camera.R, T=T_neg)
        cam_pos = CameraExtrinsics(R=camera.R, T=T_pos)

        theta = torch.deg2rad(torch.tensor(self.rotation_range_theta))
        rotation_matrix = torch.tensor(
            [
                [torch.cos(theta), 0, -torch.sin(theta), 0],
                [0, 1, 0, 0],
                [torch.sin(theta), 0, torch.cos(theta), 0],
                [0, 0, 0, 1],
            ],
            device=self.device,
        )
        RT = (cam_neg.RT @ rotation_matrix) @ cam_pos.RT
        camera.set_RT(RT)
        return camera

    def _pan_left(self, camera=None):
        if camera is None:
            camera = copy.deepcopy(self.cameras[-1])

        theta = torch.deg2rad(torch.tensor(self.rotation_range_theta))
        rotation_matrix = torch.tensor(
            [
                [torch.cos(theta), 0, -torch.sin(theta)],
                [0, 1, 0],
                [torch.sin(theta), 0, torch.cos(theta)],
            ],
            device=self.device,
        )
        R = camera.R
        R = R @ rotation_matrix
        camera.set_R(R)
        return camera

    def _pan_right(self, camera=None):
        if camera is None:
            camera = copy.deepcopy(self.cameras[-1])

        theta = torch.deg2rad(torch.tensor(-self.rotation_range_theta))
        rotation_matrix = torch.tensor(
            [
                [torch.cos(theta), 0, -torch.sin(theta)],
                [0, 1, 0],
                [torch.sin(theta), 0, torch.cos(theta)],
            ],
            device=self.device,
        )
        R = camera.R
        R = R @ rotation_matrix
        camera.set_R(R)
        return camera

    def _pull_left(self, camera=None):
        if camera is None:
            camera = copy.deepcopy(self.cameras[-1])

        camera = self._move_left(camera, magnitude=0.3)
        camera = self._pull_out(camera, magnitude=0.3)
        theta = torch.deg2rad(torch.tensor(self.rotation_range_theta))
        rotation_matrix = torch.tensor(
            [
                [torch.cos(theta), 0, -torch.sin(theta)],
                [0, 1, 0],
                [torch.sin(theta), 0, torch.cos(theta)],
            ],
            device=self.device,
        )
        R = camera.R
        R = R @ rotation_matrix
        camera.set_R(R)
        return camera

    def _pull_right(self, camera=None):
        if camera is None:
            camera = copy.deepcopy(self.cameras[-1])

        camera = self._move_right(camera, magnitude=0.3)
        camera = self._pull_out(camera, magnitude=0.3)
        theta = torch.deg2rad(torch.tensor(-self.rotation_range_theta))
        rotation_matrix = torch.tensor(
            [
                [torch.cos(theta), 0, -torch.sin(theta)],
                [0, 1, 0],
                [torch.sin(theta), 0, torch.cos(theta)],
            ],
            device=self.device,
        )
        R = camera.R
        R = R @ rotation_matrix
        camera.set_R(R)
        return camera

    def transform_all_cam_to_current_cam(self, index=-1):
        """Transform all self.cameras such that the current camera is at the origin."""

        if self.cameras != []:
            inv_current_camera_RT = torch.inverse(self.cameras[index].RT)

            for cam in self.cameras:
                cam_RT = cam.RT
                new_cam_RT = cam_RT @ inv_current_camera_RT
                cam.set_RT(new_cam_RT)

    def generate_keyframe_cameras(self):
        layout_methods = {
            "push_in": self._push_in,
            "pull_out": self._pull_out,
            "move_left": lambda: self._move_left(magnitude=0.5),
            "move_right": lambda: self._move_right(magnitude=0.5),
            "orbit_left": self._orbit_left,
            "orbit_right": self._orbit_right,
            "pan_left": self._pan_left,
            "pan_right": self._pan_right,
            # 'pull_left': self._pull_left,
            # 'pull_right': self._pull_right
        }
        try:
            next_camera = layout_methods[self.layout]()
        except:
            raise ValueError("Invalid layout:", self.layout)

        current_camera = copy.deepcopy(next_camera)
        self.cameras.append(current_camera)

        return

    def generate_interp_cameras(self):
        self.cameras_interp.append(copy.deepcopy(self.cameras[0]))
        for kf1_camera, kf2_camera in zip(self.cameras[:-1], self.cameras[1:]):
            cameras = self.interp_cameras(kf1_camera, kf2_camera)
            self.cameras_interp.extend(cameras)
            self.cameras_interp.append(copy.deepcopy(kf2_camera))

    def generate_cameras(self, camera_path, save_path, verbose: bool = False):
        for layout in camera_path:
            self.layout = layout
            self.transform_all_cam_to_current_cam()
            self.generate_keyframe_cameras()

        # transform all cameras so that the first camera is at the origin
        self.transform_all_cam_to_current_cam(index=0)
        self.generate_interp_cameras()
        cameras_list, cameras_interp_list = self.save_cameras(
            save_path, verbose=verbose
        )
        self.clear()

        return np.array(cameras_list), np.array(cameras_interp_list)

    def save_cameras(self, output_root, verbose: bool = False):
        data = {"focal_length": self.focal_length}
        data["scale"] = self.camera_speed

        matrices = []
        for camera in self.cameras:
            extrinsics = convert_extrinsics_blender(camera)
            matrices.append(extrinsics)
        cameras_list = torch.cat(matrices, dim=0).cpu().numpy().tolist()
        data["cameras"] = cameras_list

        matrices = []
        for camera in self.cameras_interp:
            extrinsics = convert_extrinsics_blender(camera)
            matrices.append(extrinsics)
        cameras_interp_list = torch.cat(matrices, dim=0).cpu().numpy().tolist()
        data["cameras_interp"] = cameras_interp_list

        if verbose:
            print("-- keyframe cameras saved! length: ", len(matrices))
            print("-- interp cameras saved! length: ", len(matrices))

        os.makedirs(output_root, exist_ok=True)
        output_path = f"{output_root}/camera_data.json"
        with open(output_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        return cameras_list, cameras_interp_list
