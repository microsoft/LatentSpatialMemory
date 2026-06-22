# ruff: noqa: E402
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from einops import rearrange
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Lambda
from transformers import T5EncoderModel, T5Tokenizer

project_root = Path(__file__).parent.parent
allegro_root = project_root / "thirdparty/Allegro"
sys.path.append(str(project_root.resolve().absolute()))
sys.path = [str(allegro_root.resolve().absolute())] + sys.path

from thirdparty.Allegro.allegro.models.transformers.transformer_3d_allegro_ti2v import (
    AllegroTransformerTI2V3DModel,
)
from thirdparty.Allegro.allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from thirdparty.Allegro.allegro.pipelines.data_process import (
    CenterCropResizeVideo,
    ToTensorVideo,
)
from thirdparty.Allegro.allegro.pipelines.pipeline_allegro_ti2v import (
    AllegroTI2VPipeline,
)


def preprocess_images(first_frame, last_frame, height, width, device, dtype):
    norm_fun = Lambda(lambda x: 2.0 * x - 1.0)
    transform = transforms.Compose(
        [ToTensorVideo(), CenterCropResizeVideo((height, width)), norm_fun]
    )
    images = []
    if first_frame is not None and len(first_frame.strip()) != 0:
        print("first_frame:", first_frame)
        images.append(first_frame)
    else:
        print("ERROR: First frame must be provided in Allegro-TI2V!")
        raise NotImplementedError
    if last_frame is not None and len(last_frame.strip()) != 0:
        print("last_frame:", last_frame)
        images.append(last_frame)

    if len(images) == 1:  # first frame as condition
        print("Video generation with given first frame.")
        conditional_images_indices = [0]
    elif len(images) == 2:  # first&last frames as condition
        print("Video generation with given first and last frame.")
        conditional_images_indices = [0, -1]
    else:
        print("ERROR: Only support 1 or 2 conditional images!")
        raise NotImplementedError

    try:
        conditional_images = [Image.open(image).convert("RGB") for image in images]
        conditional_images = [
            torch.from_numpy(np.copy(np.array(image))) for image in conditional_images
        ]
        conditional_images = [
            rearrange(image, "h w c -> c h w").unsqueeze(0)
            for image in conditional_images
        ]
        conditional_images = [
            transform(image).to(device=device, dtype=dtype)
            for image in conditional_images
        ]
    except Exception as e:
        print("Error when loading images")
        print(f"condition images are {images}")
        raise e

    return dict(
        conditional_images=conditional_images,
        conditional_images_indices=conditional_images_indices,
    )


class Allegro:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        guidance_scale: float = 8,
        num_sampling_steps: int = 100,
        seed: int = 123,
        generation_type: Literal["i2v", "t2v"] = "i2v",
    ):
        self.model_name = model_name
        self.generation_type = generation_type
        assert self.generation_type == "i2v"
        model_path = Path(model_path)
        # vae have better formance in float32
        vae = AllegroAutoencoderKL3D.from_pretrained(
            model_path / "vae",
            torch_dtype=torch.float32,
        ).cuda()
        vae.eval()

        text_encoder = T5EncoderModel.from_pretrained(
            model_path / "text_encoder",
            torch_dtype=torch.bfloat16,
        ).cuda()
        text_encoder.eval()

        tokenizer = T5Tokenizer.from_pretrained(model_path / "tokenizer")

        scheduler = EulerAncestralDiscreteScheduler()

        transformer = AllegroTransformerTI2V3DModel.from_pretrained(
            model_path / "transformer",
            torch_dtype=torch.bfloat16,
        ).cuda()
        transformer.eval()

        self.allegro_ti2v_pipeline = AllegroTI2VPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        ).to("cuda")

        self.negative_prompt = """
        nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
        low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
        """

        self.num_sampling_steps = num_sampling_steps
        self.guidance_scale = guidance_scale
        self.seed = seed

    def generate_video(self, prompt: str, image_path: str | None):
        pre_results = preprocess_images(
            image_path,
            "",
            height=720,
            width=1280,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        out_video = self.allegro_ti2v_pipeline(
            prompt,
            negative_prompt=self.negative_prompt,
            conditional_images=pre_results["conditional_images"],
            conditional_images_indices=pre_results["conditional_images_indices"],
            num_frames=88,
            height=720,
            width=1280,
            num_inference_steps=self.num_sampling_steps,
            guidance_scale=self.guidance_scale,
            max_sequence_length=512,
            generator=torch.Generator(device="cuda:0").manual_seed(self.seed),
        ).video[0]

        out_video = out_video / 255.0
        out_video = out_video.permute(0, 3, 1, 2)
        return out_video
