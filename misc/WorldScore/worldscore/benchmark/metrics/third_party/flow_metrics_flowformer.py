from typing import List
import torch
import cv2
import numpy as np

from configs.submissions import get_cfg
from core.utils import frame_utils
from core.utils.utils import InputPadder
from core.FlowFormer import build_flowformer

from worldscore.benchmark.metrics.base_metrics import BaseMetric

TRAIN_SIZE = [432, 960]


def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1] 

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size

def prepare_image(fn1, fn2, keep_size):
    print(f"preparing image...")
    print(f"fn = {fn1}")

    image1 = frame_utils.read_gen(fn1)
    image2 = frame_utils.read_gen(fn2)
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

    return image1, image2

def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    return model

def generate_pairs(img_list):
    img_pairs = []
    seq_len = len(img_list)
    for idx in range(seq_len - 1):
        img1 = img_list[idx]
        img2 = img_list[idx+1]
        img_pairs.append((img1, img2))
    return img_pairs

   
class OpticalFlowMetric(BaseMetric):
    """
    
    Using the median of estimated optical-flow to measure the motion magnitude.
    
    Optical-flow estimation -- FlowFormer++, proposed by

    FlowFormer++: Masked Cost Volume Autoencoding for Pretraining Optical Flow Estimation.
    Xiaoyu Shi*, Zhaoyang Huang*, Dasong Li, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, Hongsheng Li
    CVPR 2023.

    Ref url: https://github.com/XiaoyuShi97/FlowFormerPlusPlus
    
    RANGE: [0, ~] higher the better
    """
    
    def __init__(self) -> None:
        super().__init__()
        model = build_model()
        self._model = model.to(self._device)      
    
    def _compute_flow(self, image1, image2):
        print(f"computing flow...")
        image1, image2 = image1[None].to(self._device), image2[None].to(self._device)
        
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        with torch.amp.autocast(device_type="cuda"):
            flow_pre, _ = self._model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

        return flow
            
    def _compute_scores(
        self, 
        rendered_images: List[str],
    ) -> float:

        img_pairs = generate_pairs(rendered_images)
        scores = []
        
        with torch.no_grad():
            for img_pair in img_pairs:
                fn1, fn2 = img_pair
                print(f"processing {fn1}, {fn2}...")
                image1, image2 = prepare_image(fn1, fn2, keep_size=True)
                flow = self._compute_flow(image1, image2)
                flow_magnitude = np.sqrt((flow[..., 0] ** 2 + flow[..., 1] ** 2))
                median_flow = float(torch.from_numpy(flow_magnitude).median().item())
                scores.append(median_flow)
            
        score = sum(scores) / len(scores)
        return score