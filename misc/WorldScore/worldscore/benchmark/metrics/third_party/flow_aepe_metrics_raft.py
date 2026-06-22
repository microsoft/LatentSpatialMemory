from typing import List
import torch
import numpy as np
import argparse
from PIL import Image

from core.raft import RAFT
from core.utils.utils import InputPadder

from worldscore.benchmark.metrics.base_metrics import BaseMetric


def compute_epe(flow1, flow2, crop=30, k=1.0, error_threshold=5.0):
    """
    Calculate the End-Point Error (EPE) between two flow fields.

    Parameters:
    - flow1: Flow from image1 to image2, shape [H, W, 2].
    - flow2: Flow from image2 to image1, shape [H, W, 2].
    - crop: The size of pixels to crop for H and W.
    - k: The coefficient for the activation function.

    Returns:
    - epe: End-Point Error, a scalar value.
    """
    
    H, W, _ = flow1.shape
    crop_size_H = H - crop
    crop_size_W = W - crop
    start_x = (W - crop_size_W) // 2
    start_y = (H - crop_size_H) // 2
    
    # Crop the flow fields to the central crop_size_H x crop_size_W region
    flow1_cropped = flow1[start_y:start_y + crop_size_H, start_x:start_x + crop_size_W, :]
    flow2_cropped = flow2[start_y:start_y + crop_size_H, start_x:start_x + crop_size_W, :]
    
    # Create a grid of coordinates (x, y)
    y_coords, x_coords = np.meshgrid(np.arange(crop_size_H), np.arange(crop_size_W), indexing='ij')

    # Coordinates after applying flow1 (warping points from image1 to image2)
    warped_x1 = x_coords + flow1_cropped[..., 0]
    warped_y1 = y_coords + flow1_cropped[..., 1]

    # Warp back using flow2 from the warped positions in image2
    # First, round the warped positions and clip to image boundaries
    warped_x1_rounded = np.clip(np.round(warped_x1).astype(int), 0, crop_size_W - 1)
    warped_y1_rounded = np.clip(np.round(warped_y1).astype(int), 0, crop_size_H - 1)

    # Get the corresponding flow2 values at the new positions in image2
    flow2_at_warped = flow2_cropped[warped_y1_rounded, warped_x1_rounded, :]
    
    # Get corresponding flow from flow2 (warping back from image2 to image1)
    warped_back_x1 = warped_x1 + flow2_at_warped[..., 0]
    warped_back_y1 = warped_y1 + flow2_at_warped[..., 1]

    # Compute the End-Point Error (EPE)
    epe = np.sqrt((warped_back_x1 - x_coords) ** 2 + (warped_back_y1 - y_coords) ** 2)

    failure_mask = (epe > error_threshold).astype(np.uint8)
    # Average EPE across all pixels
    avg_epe = np.mean(epe)

    return avg_epe, failure_mask
   
class OpticalFlowAverageEndPointErrorMetric(BaseMetric):
    """
    
    Using estimated optical-flow between two consecutive frames to calculate verage end-point-error.
    
    Optical-flow estimation -- RAFT
    
    RANGE: [0, ~] lower the better
    """
    
    def __init__(self) -> None:
        super().__init__()
        
        args = {
            "model": "worldscore/benchmark/metrics/checkpoints/models/raft-things.pth",
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False,
        }
        args = argparse.Namespace(**args)
        
        # load model
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))
        model = model.module
        model.to(self._device)
        model.eval()
        self._model = model
        self._args = args    
    
    def load_image(self, imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self._device)
    
    def _compute_flow(self, image1, image2):
        print(f"computing flow...")
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        with torch.amp.autocast(device_type="cuda"):
            _, flow_up = self._model(image1, image2, iters=20, test_mode=True)

        flow = flow_up.cpu().numpy().squeeze().transpose(1, 2, 0)

        return flow
            
    def _compute_scores(
        self, 
        rendered_images: List[str],
    ) -> float:

        scores = []
        
        with torch.no_grad():
            images = rendered_images
            
            for i, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
                image1 = self.load_image(imfile1)
                image2 = self.load_image(imfile2)
                
                flow1 = self._compute_flow(image1, image2)
                flow2 = self._compute_flow(image2, image1)
                epe_score, failure_mask = compute_epe(flow1, flow2)
                scores.append(epe_score)
            
        score = sum(scores) / len(scores)
        return score.item()