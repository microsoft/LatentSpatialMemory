import sys
from pathlib import Path
from typing import Literal

import structlog
import torch
from diffusers import DiffusionPipeline

# Get  imports for vchitect
# TODO there has to be a better way than this.
project_root = Path(__file__).parent.parent
vchitect_root = project_root / "thirdparty/Vchitect2"
sys.path.append(str(vchitect_root.resolve().absolute()))

from thirdparty.Vchitect2.models.pipeline import VchitectXLPipeline  # noqa: E402

logger = structlog.get_logger(__file__)


class Vchitect:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        generation_type: Literal["t2v", "i2v"] = "t2v",
        num_frames: int = 40,
        width: int = 768,
        height: int = 432,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 100,
    ):
        assert generation_type == "t2v"
        self.pipe = VchitectXLPipeline(model_path, device="cuda")
        self.model_name = model_name
        self.generation_type = generation_type
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        self.num_frames = num_frames

    def generate_video(self, prompt: str, image_path: str | None = None):
        assert image_path is None, "Vchitect is a text-to-video model."
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            video = self.pipe(
                prompt,
                negative_prompt="",
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                width=self.width,  # 768,
                height=self.height,  # 432,
                frames=self.num_frames,
            )
            return video
