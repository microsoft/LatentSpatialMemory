from typing import List
import torch
import cv2
import numpy as np

from worldscore.benchmark.metrics.base_metrics import BaseMetric

import worldscore.benchmark.metrics.third_party.VFIMamba.config as cfg
from worldscore.benchmark.metrics.third_party.VFIMamba.Trainer_finetune import Model
from worldscore.benchmark.metrics.third_party.VFIMamba.benchmark.utils.padder import InputPadder

from worldscore.benchmark.metrics.torchmetrics.lpips_metrics import LearnedPerceptualImagePatchSimilarityMetric
# from worldscore.benchmark.metrics.torchmetrics.psnr_metrics import PeakSignalNoiseRatioMetric
from worldscore.benchmark.metrics.torchmetrics.ssim_metrics import StructuralSimilarityIndexMeasureMetric

'''==========Model setting=========='''
TTA = True
cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
    F = 32,
    depth = [2, 2, 2, 3, 3]
)

class MotionSmoothnessMetric(BaseMetric):
    """
    add description here
    """
    
    def __init__(self) -> None:
        super().__init__()
        model = Model(-1)
        model.load_model()
        model.eval()
        model.device()  
        self._model = model
        self._ssim_metric = StructuralSimilarityIndexMeasureMetric()
        self._lpips_metric = LearnedPerceptualImagePatchSimilarityMetric()
        # self._psnr_metric = PeakSignalNoiseRatioMetric()
        
    def _compute_scores(
        self, 
        rendered_images: List[str],
    ) -> float:
        scores_ssim = []
        scores_lpips = []
        scores_mse = []
        even_images = rendered_images[::2]
        odd_images = rendered_images[1::2]
        
        print(f'=========================Start Interpolating=========================')
        for i, (image1, image2) in enumerate(zip(even_images[:-1], even_images[1:])):
            odd_image = odd_images[i]
            mid_pred = cv2.imread(odd_image)
            I0 = cv2.imread(image1)
            I2 = cv2.imread(image2) 

            I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
            I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

            padder = InputPadder(I0_.shape, divisor=32)
            I0_, I2_ = padder.pad(I0_, I2_)

            mid = (padder.unpad(self._model.inference(I0_, I2_, True, TTA=TTA, fast_TTA=TTA, scale=0.0))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            
            ssim_score = self._ssim_metric._compute_scores(mid, mid_pred)
            lpips_score = self._lpips_metric._compute_scores(mid, mid_pred)
            # psnr_score = self._psnr_metric._compute_scores(mid, mid_pred)
            mse_score = np.mean((mid - mid_pred) ** 2)
            
            scores_ssim.append(ssim_score)
            scores_lpips.append(lpips_score)
            scores_mse.append(mse_score)
        print(f'=========================Done=========================')
        
        score_ssim = sum(scores_ssim) / len(scores_ssim)
        score_lpips = sum(scores_lpips) / len(scores_lpips)
        score_mse = sum(scores_mse) / len(scores_mse)
        return (score_mse, score_ssim, score_lpips)
