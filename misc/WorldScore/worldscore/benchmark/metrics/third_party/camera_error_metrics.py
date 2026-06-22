from typing import List
import torch
import numpy as np
import cv2
import argparse
from lietorch import SE3
from tqdm import tqdm
from droid import Droid
import cvxpy as cp

from worldscore.benchmark.metrics.base_metrics import BaseMetric

def image_stream(image_list, stride, calib):
    """ image generator """

    fx, fy, cx, cy = calib

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = image_list[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(imfile)

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((512 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((512 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics

def compare_rotations(R1, R2):
    cos_err = (torch.bmm(R1, R2.transpose(1, 2))[:, torch.arange(3), torch.arange(3)].sum(dim=-1) - 1) / 2
    cos_err[cos_err > 1] = 1
    cos_err[cos_err < -1] = -1
    return cos_err.acos() * 180 / np.pi

def get_cameras_accuracy(pred_Rs, gt_Rs, pred_ts, gt_ts):
    ''' Align predicted pose to gt pose and print cameras accuracy'''

    # find rotation
    d = pred_Rs.shape[-1]
    n = pred_Rs.shape[0]

    R_opt = torch.diag(torch.tensor([1., 1, 1.], device=gt_Rs.device)).double()
    R_fixed = torch.bmm(R_opt.repeat(n, 1, 1), pred_Rs)
    
    try:
        c_opt = cp.Variable()
        constraints = []
        obj = cp.Minimize(cp.sum(
            cp.norm(gt_ts.numpy() - (c_opt * pred_ts.numpy()), axis=1)))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        t_fixed = c_opt.value * pred_ts.numpy()
    except Exception as e:
        print(f"Optimization failed, using fallback method: {str(e)}")
        # Fallback: Use simple scaling based on mean distances
        gt_distances = np.linalg.norm(gt_ts.numpy(), axis=1).mean()
        pred_distances = np.linalg.norm(pred_ts.numpy(), axis=1).mean()
        scale = gt_distances / (pred_distances + 1e-8)
        if scale != 0:
            t_fixed = scale * pred_ts.numpy()
        else:
            t_fixed = pred_ts.numpy()
        
    # Calculate transaltion error
    t_error = np.linalg.norm(t_fixed - gt_ts.numpy(), axis=-1)
    t_score = np.mean(t_error).item()
    
    # Calculate rotation error
    R_error = compare_rotations(R_fixed, gt_Rs)
    R_error = R_error.numpy()
    
    r_score = np.mean(R_error).item()
    print(f"Rotation error: {r_score}")
    print(f"Translation error: {t_score}")
    return (r_score, t_score)


class CameraErrorMetric(BaseMetric):
    """
    
    return: (R_score, T_score)
    
    Range: [0, 1] higher the better
    """
    
    def __init__(self) -> None:
        super().__init__()
        args = {
            't0': 0,
            'stride': 1,
            'weights': "./worldscore/benchmark/metrics/checkpoints/droid.pth",
            'buffer': 512,
            'beta': 0.3,
            'filter_thresh': 0.01,
            'warmup': 8,
            'keyframe_thresh': 4.0,
            'frontend_thresh': 16.0,
            'frontend_window': 25,
            'frontend_radius': 2,
            'frontend_nms': 1,
            'backend_thresh': 22.0,
            'backend_radius': 2,
            'backend_nms': 3,
            # need high resolution depths
            'upsample': True,
            'stereo': False,
            'calib': [500., 500., 256., 256.]
        }
        args = argparse.Namespace(**args)
        
        self._args = args
        self.droid = None
        try:
            torch.multiprocessing.set_start_method('spawn')
        except Exception as e:
            print(f"Error setting start method: {e}")
    
    def _compute_scores(
        self, 
        rendered_images: List[str],
        cameras_gt: torch.Tensor,
        scale: float = 1.0
    ) -> float:
        
        for (t, image, intrinsics) in tqdm(image_stream(rendered_images, self._args.stride, self._args.calib)):
            if t < self._args.t0:
                continue

            if self.droid is None:
                self._args.image_size = [image.shape[2], image.shape[3]]
                self.droid = Droid(self._args)
            self.droid.track(t, image, intrinsics=intrinsics)

        traj_est, _ = self.droid.terminate(image_stream(rendered_images, self._args.stride, self._args.calib))
        cameras_pred = SE3(torch.as_tensor(traj_est)).matrix()
        
        to_blender = torch.diag(torch.tensor([-1., 1., 1., 1.], device=cameras_pred.device))
        cameras_pred = to_blender @ cameras_pred
        
        to_blender = torch.diag(torch.tensor([1., 1., -1., 1.], device=cameras_pred.device))
        w2c_cameras_pred = torch.inverse(cameras_pred)
        w2c_cameras_pred = to_blender @ w2c_cameras_pred
        cameras_pred = torch.inverse(w2c_cameras_pred)
        
        pred_Rs = cameras_pred[:, :3, :3].cpu().double()
        pred_Ts = cameras_pred[:, :3, 3].cpu().double()
    
        gt_Rs = cameras_gt[:, :3, :3].cpu().double()
        gt_Ts = cameras_gt[:, :3, 3].cpu().double() / scale
        camera_error = get_cameras_accuracy(pred_Rs, gt_Rs, pred_Ts, gt_Ts)

        self.droid = None
        return camera_error