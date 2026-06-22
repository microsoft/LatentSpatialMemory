# ruff: noqa: E402
import sys
from pathlib import Path
from typing import Literal

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from omegaconf import OmegaConf
from transformers import (
    BertModel,
    BertTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5Tokenizer,
)

# Get EasyAnimate imports
# TODO there has to be a better way than this.
project_root = Path(__file__).parent.parent
easyanimate_root = project_root / "thirdparty/EasyAnimate"
sys.path.append(str(project_root.resolve().absolute()))
sys.path = [str(easyanimate_root.resolve().absolute())] + sys.path

from thirdparty.EasyAnimate.easyanimate.models import (
    name_to_autoencoder_magvit,
    name_to_transformer3d,
)
from thirdparty.EasyAnimate.easyanimate.pipeline.pipeline_easyanimate_inpaint import (
    EasyAnimateInpaintPipeline,
)
from thirdparty.EasyAnimate.easyanimate.pipeline.pipeline_easyanimate_multi_text_encoder_inpaint import (  # noqa: E501
    EasyAnimatePipeline_Multi_Text_Encoder_Inpaint,
)
from thirdparty.EasyAnimate.easyanimate.utils.fp8_optimization import (
    convert_weight_dtype_wrapper,
)
from thirdparty.EasyAnimate.easyanimate.utils.utils import get_image_to_video_latent


class EasyAnimate:
    def __init__(
        self,
        model_name: str,
        sample_size: list[int],
        video_length: int,
        fps: int,
        generation_type: Literal["t2v", "i2v"],
        guidance_scale: float = 6.0,
        seed: int = 43,
        num_inference_steps: int = 50,
        config_path: str = (
            "thirdparty/EasyAnimate/config/easyanimate_video_v5_magvit_multi_text_encoder.yaml"
        ),
        model_path: str = "world_generators/checkpoints/easyanimate",
        sampler_name: str = "DDIM",
        GPU_memory_mode: str = "model_cpu_offload",
    ):
        weight_dtype = torch.bfloat16

        self.model_name = model_name
        self.sample_size = sample_size
        self.video_length = video_length
        self.fps = fps
        self.generation_type = generation_type
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.num_inference_steps = num_inference_steps

        config = OmegaConf.load(config_path)

        # Get Transformer
        Choosen_Transformer3DModel = name_to_transformer3d[
            config["transformer_additional_kwargs"].get(
                "transformer_type", "Transformer3DModel"
            )
        ]

        transformer_additional_kwargs = OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        )
        if weight_dtype == torch.float16:
            transformer_additional_kwargs["upcast_attention"] = True

        transformer = Choosen_Transformer3DModel.from_pretrained_2d(
            model_path,
            subfolder="transformer",
            transformer_additional_kwargs=transformer_additional_kwargs,
            torch_dtype=torch.float8_e4m3fn
            if GPU_memory_mode == "model_cpu_offload_and_qfloat8"
            else weight_dtype,
            low_cpu_mem_usage=True,
        )

        # Get Vae
        Choosen_AutoencoderKL = name_to_autoencoder_magvit[
            config["vae_kwargs"].get("vae_type", "AutoencoderKL")
        ]
        vae = Choosen_AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            vae_additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
        ).to(weight_dtype)

        if (
            config["vae_kwargs"].get("vae_type", "AutoencoderKL")
            == "AutoencoderKLMagvit"
            and weight_dtype == torch.float16
        ):
            vae.upcast_vae = True

        if config["text_encoder_kwargs"].get("enable_multi_text_encoder", False):
            tokenizer = BertTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            tokenizer_2 = T5Tokenizer.from_pretrained(
                model_path, subfolder="tokenizer_2"
            )
        else:
            tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
            tokenizer_2 = None

        if config["text_encoder_kwargs"].get("enable_multi_text_encoder", False):
            text_encoder = BertModel.from_pretrained(
                model_path, subfolder="text_encoder", torch_dtype=weight_dtype
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                model_path, subfolder="text_encoder_2", torch_dtype=weight_dtype
            )
        else:
            text_encoder = T5EncoderModel.from_pretrained(
                model_path, subfolder="text_encoder", torch_dtype=weight_dtype
            )
            text_encoder_2 = None

        if transformer.config.in_channels != vae.config.latent_channels and config[
            "transformer_additional_kwargs"
        ].get("enable_clip_in_inpaint", True):
            clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_path, subfolder="image_encoder"
            ).to("cuda", weight_dtype)
            clip_image_processor = CLIPImageProcessor.from_pretrained(
                model_path, subfolder="image_encoder"
            )
        else:
            clip_image_encoder = None
            clip_image_processor = None

        # Get Scheduler
        Choosen_Scheduler = {
            "Euler": EulerDiscreteScheduler,
            "Euler A": EulerAncestralDiscreteScheduler,
            "DPM++": DPMSolverMultistepScheduler,
            "PNDM": PNDMScheduler,
            "DDIM": DDIMScheduler,
        }[sampler_name]

        scheduler = Choosen_Scheduler.from_pretrained(model_path, subfolder="scheduler")
        if config["text_encoder_kwargs"].get("enable_multi_text_encoder", False):
            pipeline = EasyAnimatePipeline_Multi_Text_Encoder_Inpaint.from_pretrained(
                model_path,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                vae=vae,
                transformer=transformer,
                scheduler=scheduler,
                torch_dtype=weight_dtype,
                clip_image_encoder=clip_image_encoder,
                clip_image_processor=clip_image_processor,
            )
        else:
            pipeline = EasyAnimateInpaintPipeline.from_pretrained(
                model_path,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                vae=vae,
                transformer=transformer,
                scheduler=scheduler,
                torch_dtype=weight_dtype,
                clip_image_encoder=clip_image_encoder,
                clip_image_processor=clip_image_processor,
            )
        if GPU_memory_mode == "sequential_cpu_offload":
            pipeline.enable_sequential_cpu_offload()
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            pipeline.enable_model_cpu_offload()
            convert_weight_dtype_wrapper(transformer, weight_dtype)
        else:
            pipeline.enable_model_cpu_offload()

        self.vae = vae
        self.pipeline = pipeline

    def generate_video(self, prompt: str, image_path: str | None):
        video_length = self.video_length

        validation_image_start = image_path
        validation_image_end = None

        if self.vae.cache_mag_vae:
            video_length = (
                int(
                    (video_length - 1)
                    // self.vae.mini_batch_encoder
                    * self.vae.mini_batch_encoder
                )
                + 1
                if video_length != 1
                else 1
            )
        else:
            video_length = (
                int(
                    video_length
                    // self.vae.mini_batch_encoder
                    * self.vae.mini_batch_encoder
                )
                if video_length != 1
                else 1
            )
        input_video, input_video_mask, clip_image = get_image_to_video_latent(
            validation_image_start,
            validation_image_end,
            video_length=self.video_length,
            sample_size=self.sample_size,
        )

        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                video_length=video_length,
                negative_prompt="",
                height=self.sample_size[0],
                width=self.sample_size[1],
                generator=torch.Generator(device="cuda").manual_seed(self.seed),
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                video=input_video,
                mask_video=input_video_mask,
                clip_image=clip_image,
            ).videos

        sample = sample.squeeze().permute(1, 0, 2, 3)

        return sample
