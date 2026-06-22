"""
Misc utils.
"""

import numpy as np
import torch
from torchvision.utils import make_grid

import wandb


def nymeria_worker_init_fn(worker_id):
    """Worker initialization function for Nymeria dataset with LMDB.

    Reopens LMDB environments in worker processes to avoid sharing across processes.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "reopen_envs"):
        worker_info.dataset.reopen_envs()


def prepare_for_saving(tensor, fps=16, caption=None):
    # Convert range [-1, 1] to [0, 1]
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1).detach()
    if tensor.ndim == 4:
        # Assuming it's an image and has shape [batch_size, 3, height, width]
        tensor = make_grid(tensor, 4, padding=0, normalize=False)
        return wandb.Image(
            (tensor * 255).cpu().permute(1, 2, 0).numpy().astype(np.uint8),
            caption=caption,
        )
    elif tensor.ndim == 5:
        # Assuming it's a video and has shape [batch_size, num_frames, 3, height, width]
        return wandb.Video(
            (tensor * 255).cpu().numpy().astype(np.uint8),
            fps=fps,
            format="mp4",
            caption=caption,
        )
    else:
        raise ValueError(
            "Unsupported tensor shape for saving. Expected 4D (image) or 5D (video) tensor."
        )


def print_gpu_memory(label: str = "") -> None:
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA not available, skipping GPU memory report.")
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(
        f"[{label}] GPU memory — allocated: {allocated:.3f} GB, reserved: {reserved:.3f} GB"
    )
