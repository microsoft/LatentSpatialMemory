import warnings
from glob import glob
from typing import Optional, Tuple, Union

import numpy as np
import torch
from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.core.vram import (
    AutoWrappedLinear,
    AutoWrappedModule,
    enable_vram_management,
)
from diffsynth.diffusion import FlowMatchScheduler
from diffsynth.diffusion.base_pipeline import (
    BasePipeline,
    PipelineUnit,
    PipelineUnitRunner,
)
from einops import repeat
from PIL import Image
from tqdm import tqdm

from .flow_match import FlowUniPCMultistepScheduler
from .lora import GeneralLoRALoaderWithUnload
from .model_manager import ModelManager
from .wan_prompter import WanPrompter
from .wan_video_dit import RMSNorm, WanModel, sinusoidal_embedding_1d
from .wan_video_image_encoder import WanImageEncoder
from .wan_video_text_encoder import T5LayerNorm, T5RelativeEmbedding, WanTextEncoder
from .wan_video_vace import VaceWanModel
from .wan_video_vae import CausalConv3d, RMS_norm, Upsample, WanVideoVAE

WanAutoCastLayerNorm = AutoWrappedModule


def expand_local_model_path(path: str | list[str]) -> str | list[str]:
    if not isinstance(path, str) or not any(char in path for char in "*?["):
        return path
    matches = sorted(glob(path))
    if not matches:
        raise FileNotFoundError(f"Model path pattern did not match any files: {path}")
    return matches[0] if len(matches) == 1 else matches


class WanVideoPipeline(BasePipeline):
    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.bfloat16,
        tokenizer_path=None,
        units=[],
        extract_vae_latents_mode=False,
        use_reentrant=False,
    ):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.vace: VaceWanModel = None
        self.in_iteration_models = ("dit", "vace")
        self.in_iteration_models_2 = ("dit2", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.units = units

        self.model_fn = model_fn_wan_video
        self.extract_vae_latents_mode = extract_vae_latents_mode
        self.use_reentrant = use_reentrant
        self.lora_loaded = False

    def load_lora(self, module, lora_path=None, lora_state_dict=None, alpha=1):
        if self.lora_loaded:
            print("LoRA is already loaded, skip loading")
            return
        loader = GeneralLoRALoaderWithUnload(
            torch_dtype=self.torch_dtype, device=self.device
        )
        if lora_path is not None:
            lora = load_state_dict(
                lora_path, torch_dtype=self.torch_dtype, device=self.device
            )
        elif lora_state_dict is not None:
            lora = lora_state_dict
        else:
            raise ValueError("Either lora_path or lora_state_dict must be provided")
        loader.load(module, lora, alpha=alpha)
        self.lora_loaded = True

    def unload_lora(self, module, lora_path=None, lora_state_dict=None, alpha=1):
        if not self.lora_loaded:
            print("LoRA is not loaded, skip unloading")
            return
        loader = GeneralLoRALoaderWithUnload(
            torch_dtype=self.torch_dtype, device=self.device
        )
        if lora_path is not None:
            lora = load_state_dict(
                lora_path, torch_dtype=self.torch_dtype, device=self.device
            )
        elif lora_state_dict is not None:
            lora = lora_state_dict
        else:
            raise ValueError("Either lora_path or lora_state_dict must be provided")
        loader.unload(module, lora, alpha=alpha)
        self.lora_loaded = False

    def enable_vram_management(
        self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5
    ):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                vram_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                vram_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                vram_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                vram_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                vram_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                vram_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[Tuple[dict, ModelConfig]] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"
        ),
        redirect_common_files: bool = True,
        **pipeline_kwargs,
    ):
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config_info in model_configs:
                if isinstance(model_config_info, tuple):
                    _, model_config = model_config_info
                else:
                    model_config = model_config_info

                if (
                    model_config.origin_file_pattern is None
                    or model_config.model_id is None
                ):
                    continue
                if (
                    model_config.origin_file_pattern in redirect_dict
                    and model_config.model_id
                    != redirect_dict[model_config.origin_file_pattern]
                ):
                    print(
                        f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection."
                    )
                    model_config.model_id = redirect_dict[
                        model_config.origin_file_pattern
                    ]

        pipe = WanVideoPipeline(
            device=device, torch_dtype=torch_dtype, **pipeline_kwargs
        )

        model_manager = ModelManager(strict_load=False)
        for model_config_info in model_configs:
            if isinstance(model_config_info, tuple):
                model_extra_kwargs, model_config = model_config_info
            else:
                model_extra_kwargs, model_config = {}, model_config_info
            model_config.download_if_necessary()
            model_config.path = expand_local_model_path(model_config.path)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype,
                class_overwrite_kwargs=model_extra_kwargs,
            )

        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.vace = model_manager.fetch_model("wan_video_vace")

        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        tokenizer_config.download_if_necessary()
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        return pipe

    def preprocess_image(
        self,
        image,
        torch_dtype=None,
        device=None,
        pattern="B C H W",
        min_value=-1,
        max_value=1,
    ):
        if isinstance(image, Image.Image):
            image = torch.Tensor(np.array(image, dtype=np.float32))
            raw_pattern = "H W C"
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
            raw_pattern = "H W C"
        else:
            raw_pattern = "C H W"

        image = image.to(
            dtype=torch_dtype or self.torch_dtype, device=device or self.device
        )
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(
            image, f"{raw_pattern} -> {pattern}", **({"B": 1} if "B" in pattern else {})
        )
        return image

    @torch.no_grad()
    def call_latent_inference(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        input_image: Optional[Image.Image] = None,
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        ar_hist_latents_num: Optional[int] = None,
        ref_images: Optional[list[Image.Image]] = None,
        control_video: Optional[list[Image.Image] | list[np.ndarray]] = None,
        control_score: Optional[list[Image.Image] | list[np.ndarray]] = None,
        vace_scale: Optional[float] = 1.0,
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        cfg_scale: Optional[float] = 5.0,
        switch_DiT_boundary: Optional[float] = 0.875,
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        progress_bar_cmd=tqdm,
        return_latents: Optional[bool] = False,
        sampler: Optional[str] = "uni_pc",
        verbose: Optional[bool] = True,
    ):
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        if sampler == "euler":
            self.scheduler.set_timesteps(
                num_inference_steps,
                denoising_strength=denoising_strength,
                shift=sigma_shift,
            )
            num_train_timesteps = self.scheduler.num_train_timesteps
        elif sampler == "uni_pc":
            num_train_timesteps = self.scheduler.num_train_timesteps
            self.scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                shift=1.0,
                use_dynamic_shifting=False,
            )
            self.scheduler.set_timesteps(
                num_inference_steps, device=self.device, shift=sigma_shift
            )
        inputs_posi = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "ref_images": ref_images,
            "input_video": input_video,
            "denoising_strength": denoising_strength,
            "control_video": control_video,
            "control_score": control_score,
            "vace_scale": vace_scale,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "sigma_shift": sigma_shift,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        if ref_images is not None and len(ref_images) > 0:
            ref_frame_latents = inputs_shared["latents"][
                :, :, : len(ref_images)
            ].clone()
            valid_ref_frames_num = ref_frame_latents.shape[2]
            inputs_shared["num_ref_frames"] = ref_frame_latents.shape[2]
        else:
            valid_ref_frames_num = 0
        if ar_hist_latents_num is not None and ar_hist_latents_num > 0:
            ar_hist_latents = inputs_shared["latents"][
                :, :, valid_ref_frames_num : valid_ref_frames_num + ar_hist_latents_num
            ].clone()
        for progress_id, timestep in enumerate(
            progress_bar_cmd(self.scheduler.timesteps, disable=not verbose)
        ):
            if (
                timestep.item() < switch_DiT_boundary * num_train_timesteps
                and self.dit2 is not None
                and not models["dit"] is self.dit2
            ):
                print(f"Switching to DiT2 at timestep {timestep.item()}")
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2
            timestep = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)
            timestep_for_sampler = timestep.clone()
            if ar_hist_latents_num is not None and ar_hist_latents_num > 0:
                inputs_shared["fuse_vae_embedding_in_latents"] = True
                timestep = timestep.repeat(inputs_shared["latents"].shape[2])
                timestep[: valid_ref_frames_num + ar_hist_latents_num] = 0
            noise_pred_posi = self.model_fn(
                **models, **inputs_shared, **inputs_posi, timestep=timestep
            )
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(
                    **models, **inputs_shared, **inputs_nega, timestep=timestep
                )
                noise_pred = noise_pred_nega + cfg_scale * (
                    noise_pred_posi - noise_pred_nega
                )
            else:
                noise_pred = noise_pred_posi

            if sampler == "euler":
                inputs_shared["latents"] = self.scheduler.step(
                    noise_pred,
                    self.scheduler.timesteps[progress_id],
                    inputs_shared["latents"],
                )
            elif sampler == "uni_pc":
                inputs_shared["latents"] = self.scheduler.step(
                    noise_pred,
                    timestep_for_sampler[0],
                    inputs_shared["latents"],
                    return_dict=False,
                    generator=seed_g,
                )[0]
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared[
                    "first_frame_latents"
                ]
            if ar_hist_latents_num is not None and ar_hist_latents_num > 0:
                inputs_shared["latents"][
                    :,
                    :,
                    valid_ref_frames_num : valid_ref_frames_num + ar_hist_latents_num,
                ] = ar_hist_latents
            if valid_ref_frames_num > 0:
                inputs_shared["latents"][:, :, :valid_ref_frames_num] = (
                    ref_frame_latents
                )

        if valid_ref_frames_num > 0:
            inputs_shared["latents"] = inputs_shared["latents"][
                :, :, valid_ref_frames_num:
            ]
        self.load_models_to_device(["vae"])
        latents = inputs_shared["latents"]
        video = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        if return_latents:
            return video, latents
        else:
            return video


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(
            height, width, num_frames
        )
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device")
        )

    def process(
        self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device
    ):
        length = (num_frames - 1) // 4 + 1
        shape = (
            1,
            pipe.vae.model.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class WanVideoUnit_HistVideoEmbedder(PipelineUnit):
    def __init__(self, num_hist_frames=1):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",),
        )
        self.num_hist_frames = num_hist_frames

    def process(
        self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride
    ):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video[-self.num_hist_frames :])
        input_latents = pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        latents = noise
        if input_latents.shape[2] > 0:
            latents[:, :, : input_latents.shape[2]] = input_latents
        if pipe.training:
            raise ValueError(
                "Training is not supported for WanVideoUnit_HistVideoEmbedder"
            )
        else:
            return {"latents": latents}


class WanVideoUnit_VideoEmbedLoader(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents", "noise"), onload_model_names=None)

    def process(self, pipe: WanVideoPipeline, latents, noise):
        if latents is None:
            return {"latents": noise}

        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": latents}
        else:
            latents = pipe.scheduler.add_noise(
                latents, noise, timestep=pipe.scheduler.timesteps[0]
            )
            return {"latents": latents}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self, uncond_p=0.1):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",),
        )
        self.uncond_p = uncond_p

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        if torch.rand(1) < self.uncond_p:
            prompt = ["" for _ in range(len(prompt))]
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(
            prompt, positive=positive, device=pipe.device
        )
        return {"context": prompt_emb}


class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """Encode the input image into fused VAE latents."""

    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "latents",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
                "ref_images",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_image,
        latents,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        ref_images=None,
    ):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(input_image, Image.Image):
            resized_image = input_image.resize((width, height))
        elif isinstance(input_image, np.ndarray):
            resized_image = Image.fromarray(input_image).resize((width, height))
        else:
            resized_image = Image.fromarray(
                np.array(input_image.cpu()).transpose(1, 2, 0)
            ).resize((width, height))
        image = pipe.preprocess_image(resized_image).transpose(0, 1)
        first_frame_z = pipe.vae.encode(
            [image],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if ref_images is not None and len(ref_images) > 0:
            ref_images = [
                pipe.preprocess_image(ref_image_i).transpose(0, 1)
                for ref_image_i in ref_images
            ]
            ref_z = pipe.vae.encode(
                ref_images,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            ref_z = (
                ref_z.squeeze(2).transpose(0, 1).unsqueeze(0).contiguous()
            )  # [T,C,1,H,W]->[1,C,T,H,W]
        else:
            ref_z = None
        latents[:, :, :1] = first_frame_z
        if ref_z is not None:
            latents = torch.concat([ref_z, latents], dim=2)
        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "first_frame_latents": first_frame_z,
            "ref_frame_latents": ref_z,
        }


class WanVideoUnit_RefImageEmbedderFused(PipelineUnit):
    """Encode reference images into fused VAE latents."""

    def __init__(self):
        super().__init__(
            input_params=(
                "latents",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
                "ref_images",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        latents,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        ref_images=None,
    ):
        if ref_images is not None and len(ref_images) > 0:
            ref_images = [
                pipe.preprocess_image(ref_image_i).transpose(0, 1)
                for ref_image_i in ref_images
            ]
            ref_z = pipe.vae.encode(
                ref_images,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            ref_z = (
                ref_z.squeeze(2).transpose(0, 1).unsqueeze(0).contiguous()
            )  # [T,C,1,H,W]->[1,C,T,H,W]
        else:
            ref_z = None
        latents = (
            torch.concat([ref_z, latents], dim=2) if ref_z is not None else latents
        )
        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "ref_frame_latents": ref_z,
        }


class WanVideoUnit_ControlNetAsVACEEmbedder(PipelineUnit):
    def __init__(self, default_vace_scale=1.0):
        super().__init__(
            input_params=(
                "control_video",
                "control_score",
                "vace_scale",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=None,
        )
        self.default_vace_scale = default_vace_scale

    def process(
        self,
        pipe: WanVideoPipeline,
        control_video,
        control_score,
        vace_scale,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if control_video is not None or control_score is not None:
            pipe.load_models_to_device(["vae"])
            if control_video is None:
                control_video = torch.zeros(
                    (1, 3, num_frames, height, width),
                    dtype=pipe.torch_dtype,
                    device=pipe.device,
                )
            else:
                control_video = pipe.preprocess_video(control_video)

            if control_score is None:
                control_score = torch.ones_like(control_video)
            else:
                control_score = pipe.preprocess_video(
                    control_score, min_value=0, max_value=1
                )

            control_latent = pipe.vae.encode(
                control_video,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            control_score_latent = pipe.vae.encode(
                control_score,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat(
                (control_latent, control_score_latent), dim=1
            )
            return {
                "vace_context": vace_video_latents,
                "vace_scale": vace_scale
                if vace_scale is not None
                else self.default_vace_scale,
            }
        else:
            return {
                "vace_context": None,
                "vace_scale": vace_scale
                if vace_scale is not None
                else self.default_vace_scale,
            }


def model_fn_wan_video(
    dit: WanModel,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    vace_context=None,
    vace_scale=1.0,
    num_ref_frames=0,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    fuse_vae_embedding_in_latents: bool = False,
    use_reentrant=False,
    **kwargs,
):
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        if len(timestep) == 1:
            timestep = torch.concat(
                [
                    torch.zeros(
                        (1, latents.shape[3] * latents.shape[4] // 4),
                        dtype=latents.dtype,
                        device=latents.device,
                    ),
                    torch.ones(
                        (
                            latents.shape[2] - 1,
                            latents.shape[3] * latents.shape[4] // 4,
                        ),
                        dtype=latents.dtype,
                        device=latents.device,
                    )
                    * timestep,
                ]
            ).flatten()
        else:
            timestep = timestep.unsqueeze(1).repeat(
                1, latents.shape[3] * latents.shape[4] // 4
            )
            timestep = timestep.flatten()
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep)
            .unsqueeze(0)
            .to(dtype=latents.dtype)
        )
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).to(dtype=latents.dtype)
        )
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents

    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    x, (f_all, h, w) = dit.patchify(x)

    ref_token_num = num_ref_frames * h * w
    f = f_all - num_ref_frames

    freqs = (
        torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(x.device)
    )

    if num_ref_frames > 0:
        ref_freqs = (
            torch.cat(
                [
                    dit.ref_freqs[0][-num_ref_frames:]
                    .view(num_ref_frames, 1, 1, -1)
                    .expand(num_ref_frames, h, w, -1),
                    dit.ref_freqs[1][h : 2 * h]
                    .view(1, h, 1, -1)
                    .expand(num_ref_frames, h, w, -1),
                    dit.ref_freqs[2][:w]
                    .view(1, 1, w, -1)
                    .expand(num_ref_frames, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(num_ref_frames * h * w, 1, -1)
            .to(x.device)
        )
        freqs = torch.cat([ref_freqs, freqs], dim=0)
    if vace_context is not None and vace is not None:
        vace_h = h
        vace_w = w
        freqs_vace = freqs
        vace_input_x = x
        vace_t_mod = t_mod
        vace_input_x = vace_input_x[:, ref_token_num:]
        vace_t_mod = vace_t_mod[:, ref_token_num:]
        freqs_vace = freqs_vace[ref_token_num:]
        vace_hints = vace(
            vace_input_x,
            vace_context,
            context,
            vace_t_mod,
            freqs_vace,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    context,
                    t_mod,
                    freqs,
                    use_reentrant=use_reentrant,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x,
                context,
                t_mod,
                freqs,
                use_reentrant=use_reentrant,
            )
        else:
            x = block(x, context, t_mod, freqs)
        if vace_context is not None and vace is not None:
            if block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                x[:, ref_token_num:] = (
                    x[:, ref_token_num:] + current_vace_hint * vace_scale
                )

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f_all, h, w))
    return x
