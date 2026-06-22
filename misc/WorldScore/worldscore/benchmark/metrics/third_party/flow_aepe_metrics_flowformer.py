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

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]

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
    
    Optical-flow estimation -- FlowFormer++, proposed by

    FlowFormer++: Masked Cost Volume Autoencoding for Pretraining Optical Flow Estimation.
    Xiaoyu Shi*, Zhaoyang Huang*, Dasong Li, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, Hongsheng Li
    CVPR 2023.

    Ref url: https://github.com/XiaoyuShi97/FlowFormerPlusPlus
    
    RANGE: [0, ~] lower the better
    """
    
    def __init__(self) -> None:
        super().__init__()
        model = build_model()
        self._model = model.to(self._device)      
    
    def _compute_flow(self, image1, image2):
        print(f"computing flow...")
        image_size = image1.shape[1:]
        image1, image2 = image1[None].to(self._device), image2[None].to(self._device)
        hws = compute_grid_indices(image_size)
        
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
                flow1 = self._compute_flow(image1, image2)
                flow2 = self._compute_flow(image2, image1)
                epe_score, failure_mask = compute_epe(flow1, flow2)
                scores.append(epe_score)
            
        score = sum(scores) / len(scores)
        return score.item()