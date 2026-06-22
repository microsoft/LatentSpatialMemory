# pylint: disable=R0913,R0914,C0103
"""Wrapper around CogVideoX."""

from typing import Literal

import structlog
import torch
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video

logger = structlog.getLogger()


class CogVideoX:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        generation_type: Literal["t2v", "i2v", "v2v"],
        lora_path: str = None,
        lora_rank: int = 128,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        num_frames: int = 49,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
    ):
        """
        Initialized a CogVideoX model.

        Parameters:
            - model_name (str): Model name.
            - model_path (str): The path of the pre-trained model to be used.
            - lora_path (str): The path of the LoRA weights to be used.
            - lora_rank (int): The rank of the LoRA weights.
            - num_inference_steps (int): Number of steps for the inference process. More
                steps can result in better quality.
            - guidance_scale (float): The scale for classifier-free guidance. Higher
                values can lead to better alignment with the prompt.
            - num_videos_per_prompt (int): Number of videos to generate per prompt.
            - num_frames (int): Number of frames to generate for each video.
            - dtype (torch.dtype): Data type for computation (default: torch.bfloat16).
            - generation_type (str): Type of video generation (t2v, i2v, v2v).
            - seed (int): The seed for reproducibility.
        """
        self.generation_type = generation_type
        self.num_videos_per_prompt = num_videos_per_prompt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.num_frames = num_frames
        self.use_dynamic_cfg = True

        # Load the pre-trained CogVideoX pipeline with specified precision.
        pipeline_types = {
            "i2v": CogVideoXImageToVideoPipeline,
            "t2v": CogVideoXPipeline,
            "v2v": CogVideoXVideoToVideoPipeline,
        }
        self.pipe = pipeline_types[self.generation_type].from_pretrained(
            model_path, torch_dtype=dtype
        )

        # Load LORA weights, if used.
        if lora_path:
            self.pipe.load_lora_weights(
                lora_path,
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name="test_1",
            )
            self.pipe.fuse_lora(lora_scale=1 / lora_rank)

        # 2. Set Scheduler.
        # Recommendation from CogVideo:
        #   - CogVideoXDDIMScheduler for CogVideoX-2B.
        #   - CogVideoXDPMScheduler for CogVideoX-5B / CogVideoX-5B-I2V.
        if "5b" in model_name:
            self.pipe.scheduler = CogVideoXDPMScheduler.from_config(
                self.pipe.scheduler.config, timestep_spacing="trailing"
            )
        elif "2b" in model_name:
            self.use_dynamic_cfg = False  # Repo suggests turning off from DDIM.
            self.pipe.scheduler = CogVideoXDDIMScheduler.from_config(
                self.pipe.scheduler.config, timestep_spacing="trailing"
            )
        else:
            raise ValueError("Expected a model in the cogvideox family.")

        # Move pipeline to CUDA.
        self.pipe.to("cuda")
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

    def generate_video(
        self,
        prompt: str,
        image_path: str | None,
    ):
        prompt_kws = {"prompt": prompt}
        if self.generation_type == "i2v":
            prompt_kws["image"] = load_image(image=image_path)
        elif self.generation_type == "v2v":
            prompt_kws["video"] = load_video(video=image_path)

        generated_out = self.pipe(
            **prompt_kws,
            num_videos_per_prompt=self.num_videos_per_prompt,
            num_inference_steps=self.num_inference_steps,
            num_frames=self.num_frames,
            use_dynamic_cfg=self.use_dynamic_cfg,
            guidance_scale=self.guidance_scale,
            generator=torch.Generator().manual_seed(self.seed),
        )

        generated_video = generated_out.frames[0]
        return generated_video
