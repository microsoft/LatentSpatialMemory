from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class SceneProjectionData:
    """Data class for scene projection and related geometry information."""

    scene_proj: torch.Tensor
    points_world: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    dynamic_mask_frame0: Optional[np.ndarray] = None
    poses_c2w: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    processed_size: Optional[tuple] = None
    anchor_depth_frame0: Optional[np.ndarray] = None
    anchor_frame0: Optional[np.ndarray] = None
    initial_points_world: Optional[np.ndarray] = None
    initial_colors: Optional[np.ndarray] = None
