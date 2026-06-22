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


class ReprojectionErrorMetric(BaseMetric):
    """
    
    return: Reprojection error
    
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
    ) -> float:
        
        for (t, image, intrinsics) in tqdm(image_stream(rendered_images, self._args.stride, self._args.calib)):
            if t < self._args.t0:
                continue

            if self.droid is None:
                self._args.image_size = [image.shape[2], image.shape[3]]
                self.droid = Droid(self._args)
            self.droid.track(t, image, intrinsics=intrinsics)

        traj_est, valid_errors = self.droid.terminate(image_stream(rendered_images, self._args.stride, self._args.calib))
        
        if len(valid_errors) > 0:
            mean_error = valid_errors.mean().item()

        self.droid = None
        return mean_error