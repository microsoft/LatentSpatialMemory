# pylint: disable=R0913,R0914,C0103
"""Wrapper around VideoCrafter"""

import os
import sys
from pathlib import Path
from typing import Literal, Optional

import structlog
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

# Get VideoCrafter imports
# TODO there has to be a better way than this.
project_root = Path(__file__).parent.parent
videocrafter_root = project_root / "thirdparty/VideoCrafter"
sys.path.append(str(project_root.resolve().absolute()))
sys.path.append(str(videocrafter_root.resolve().absolute()))


from thirdparty.VideoCrafter.scripts.evaluation.funcs import (  # noqa: E402
    batch_ddim_sampling,
    load_image_batch,
    load_model_checkpoint,
)
from thirdparty.VideoCrafter.utils.utils import (  # noqa: E402
    instantiate_from_config,
)

logger = structlog.getLogger()


class VideoCrafter:
    def __init__(
        self,
        model_name: str,
        config: str,
        ckpt_path: str,
        height: int,
        width: int,
        generation_type: Literal["t2v", "i2v"],
        frames: int = -1,
        fps: int = 8,
        n_samples: int = 1,
        ddim_steps: int = 50,
        ddim_eta: float = 1.0,
        unconditional_guidance_scale: float = 12.0,
        seed: int = 123,
    ):
        seed_everything(seed)

        # Load model confic
        config = OmegaConf.load(config)
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)
        model = model.cuda()
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path} not found")

        # Load mode
        self.model = load_model_checkpoint(model, ckpt_path)
        self.model.eval()

        ## sample shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("Error: image size [h,w] should be multiples of 16!")

        self.model_name = model_name
        self.height = height
        self.width = width
        self.frames = frames
        self.fps = fps
        self.generation_type = generation_type
        self.n_samples = n_samples
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.unconditional_guidance_scale = unconditional_guidance_scale

    def generate_video(self, prompt: str, image_path: Optional[str] = None):
        ## latent noise shape
        h, w = self.height // 8, self.width // 8
        frames = self.model.temporal_length if self.frames < 0 else self.frames
        channels = self.model.channels

        noise_shape = [1, channels, frames, h, w]
        fps = torch.tensor([self.fps]).to(self.model.device).long()

        prompts = [prompt]
        # prompts = batch_size * [""]
        text_emb = self.model.get_learned_conditioning(prompts)

        if self.generation_type == "t2v":
            cond = {"c_crossattn": [text_emb], "fps": fps}
        elif self.generation_type == "i2v":
            cond_images = load_image_batch([image_path], (self.height, self.width))
            cond_images = cond_images.to(self.model.device)
            img_emb = self.model.get_image_embeds(cond_images)
            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            cond = {"c_crossattn": [imtext_cond], "fps": fps}
        else:
            raise NotImplementedError

        ## inference
        batch_samples = batch_ddim_sampling(
            self.model,
            cond,
            noise_shape,
            self.n_samples,
            self.ddim_steps,
            self.ddim_eta,
            self.unconditional_guidance_scale,
        )

        batch_samples = batch_samples.detach().squeeze().cpu()
        batch_samples = torch.clamp(batch_samples.float(), -1.0, 1.0)
        batch_samples = (batch_samples + 1.0) / 2.0
        batch_samples = batch_samples.permute(1, 0, 2, 3)
        return batch_samples
