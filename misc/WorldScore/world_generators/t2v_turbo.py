# Adapted from https://github.com/luosiallen/latent-consistency-model
import sys
from pathlib import Path
from typing import Literal

import torch
from omegaconf import OmegaConf

# Get TV2TurboVC2 imports
# TODO there has to be a better way than this.
project_root = Path(__file__).parent.parent
t2vturbo_root = project_root / "thirdparty/t2v_turbo"
sys.path.append(str(t2vturbo_root.resolve().absolute()))

from pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline  # noqa: E402
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler  # noqa: E402
from utils.common_utils import load_model_checkpoint  # noqa: E402
from utils.lora import collapse_lora, monkeypatch_remove_lora  # noqa: E402
from utils.lora_handler import LoraHandler  # noqa: E402
from utils.utils import instantiate_from_config  # noqa: E402


class T2VTurbo:
    def __init__(
        self,
        model_name,
        config: str,
        model_ckpt: str,
        lora_path: str,
        generation_type: Literal["t2v", "i2v"],
        num_frames: int = 48,
        fps: int = 16,
        seed: int = 0,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 4,
    ):
        torch.manual_seed(seed)
        assert generation_type != "i2v"

        model_config = OmegaConf.load(config).pop("model", OmegaConf.create())
        pretrained_t2v = instantiate_from_config(model_config)
        pretrained_t2v = load_model_checkpoint(pretrained_t2v, model_ckpt)

        unet_config = model_config["params"]["unet_config"]
        unet_config["params"]["time_cond_proj_dim"] = 256
        unet = instantiate_from_config(unet_config)
        unet.load_state_dict(
            pretrained_t2v.model.diffusion_model.state_dict(), strict=False
        )

        # Update LORA
        use_unet_lora = True
        lora_manager = LoraHandler(
            version="cloneofsimo",
            use_unet_lora=use_unet_lora,
            save_for_webui=True,
            unet_replace_modules=["UNetModel"],
        )
        lora_manager.add_lora_to_model(
            use_unet_lora,
            unet,
            lora_manager.unet_replace_modules,
            lora_path=lora_path,
            dropout=0.1,
            r=64,
        )
        unet.eval()
        collapse_lora(unet, lora_manager.unet_replace_modules)
        monkeypatch_remove_lora(unet)

        pretrained_t2v.model.diffusion_model = unet
        scheduler = T2VTurboScheduler(
            linear_start=model_config["params"]["linear_start"],
            linear_end=model_config["params"]["linear_end"],
        )
        self.pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)
        self.pipeline = self.pipeline.to(device="cuda")

        self.model_name = model_name
        self.fps = fps
        self.num_frames = num_frames
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

    def generate_video(self, prompt: str, image_path: str | None = None):
        assert image_path is None
        video = self.pipeline(
            prompt=prompt,
            frames=self.num_frames,
            fps=self.fps,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            num_videos_per_prompt=1,
        )

        video = video.detach().squeeze().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = (video + 1.0) / 2.0
        video = video.permute(1, 0, 2, 3)
        return video
