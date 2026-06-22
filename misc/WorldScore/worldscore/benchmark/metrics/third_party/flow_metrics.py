import argparse
from typing import List
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import cv2

from core.raft import RAFT
from core.utils.utils import load_ckpt
from core.parser import parse_args

from worldscore.benchmark.metrics.base_metrics import BaseMetric


class OpticalFlowMetric(BaseMetric):
    """
    
    Using the median of estimated optical-flow to measure the motion magnitude.
    
    Optical-flow estimation -- SEA-RAFT
    
    RANGE: [0, ~] higher the better
    """
    
    def __init__(self) -> None:
        super().__init__()
        
        args = {
            "cfg": "worldscore/benchmark/metrics/third_party/SEA-RAFT/config/eval/spring-M.json",
            "path": "worldscore/benchmark/metrics/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth",
        }
        args = argparse.Namespace(**args)
        args = parse_args(args)
        
        # load model
        model = RAFT(args)
        load_ckpt(model, args.path)
        model.to(self._device)
        model.eval()
        self._model = model
        self._args = args
    
    def load_image(self, imfile):
        image = cv2.imread(imfile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image = image[None].to(self._device)
        return image

    def forward_flow(self, image1, image2):
        with torch.amp.autocast(device_type="cuda"):
            output = self._model(image1, image2, iters=self._args.iters, test_mode=True)
        flow_final = output['flow'][-1]
        info_final = output['info'][-1]
        return flow_final, info_final

    def _compute_flow(self, image1, image2):
        print(f"computing flow...")
    
        img1 = F.interpolate(image1, scale_factor=2 ** self._args.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** self._args.scale, mode='bilinear', align_corners=False)
        H, W = img1.shape[2:]
        flow, info = self.forward_flow(img1, img2)
        flow_down = F.interpolate(flow, scale_factor=0.5 ** self._args.scale, mode='bilinear', align_corners=False) * (0.5 ** self._args.scale)
        info_down = F.interpolate(info, scale_factor=0.5 ** self._args.scale, mode='area')
        
        flow = flow_down.cpu().numpy().squeeze().transpose(1, 2, 0)
        return flow
            
    def _compute_scores(
        self, 
        rendered_images: List[str],
    ) -> float:

        scores = []
        
        with torch.no_grad():
            images = rendered_images
            
            for i, (imfile1, imfile2) in tqdm(enumerate(zip(images[:-1], images[1:])), total=len(images) - 1, desc="Computing flow..."):
                image1 = self.load_image(imfile1)
                image2 = self.load_image(imfile2)
                
                flow = self._compute_flow(image1, image2)
                flow_magnitude = np.sqrt((flow[..., 0] ** 2 + flow[..., 1] ** 2))
                median_flow = float(torch.from_numpy(flow_magnitude).median().item())
                scores.append(median_flow)
            
        score = sum(scores) / len(scores)
        return score