import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything

project_root = Path(__file__).parent.parent
dynamicrafter_root = project_root / "thirdparty/DynamiCrafter"
sys.path.append(str(project_root.resolve().absolute()))
sys.path.append(str(dynamicrafter_root.resolve().absolute()))

from thirdparty.DynamiCrafter.lvdm.models.samplers.ddim import (  # noqa: E402
    DDIMSampler,
)
from thirdparty.DynamiCrafter.lvdm.models.samplers.ddim_multiplecond import (  # noqa: E402
    DDIMSampler as DDIMSampler_multicond,
)
from thirdparty.DynamiCrafter.utils.utils import instantiate_from_config  # noqa: E402


class DynamiCrafter:
    def __init__(
        self,
        model_name: str,
        generation_type: str,
        config: str,
        ckpt_path: str,
        height: int,
        width: int,
        perframe_ae: bool = False,
        video_length: int = 16,
        n_samples: int = 1,
        ddim_steps: int = 50,
        ddim_eta: float = 1.0,
        unconditional_guidance_scale: float = 7.5,
        cfg_img: float | None = None,
        frame_stride: int = 3,
        text_input: bool = False,
        multiple_cond_cfg: bool = False,
        loop: bool = False,
        interp: bool = False,
        timestep_spacing: str = "uniform",
        guidance_rescale: float = 0.0,
        seed: int = 123,
    ):
        seed_everything(seed)
        assert generation_type == "i2v"

        # Load model
        model_config = OmegaConf.load(config).pop("model", OmegaConf.create())

        # set use_checkpoint as False as when using deepspeed, it encounters an error
        # "deepspeed backend not set"
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
        model = instantiate_from_config(model_config)
        model = model.cuda()
        model.perframe_ae = perframe_ae
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path} not found")

        # Load model
        self.model = load_model_checkpoint(model, ckpt_path)
        self.model.eval()

        ## sample shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("Error: image size [h,w] should be multiples of 16!")

        self.model_name = model_name
        self.height = height
        self.width = width
        self.channels = self.model.model.diffusion_model.out_channels
        self.video_length = video_length
        self.n_samples = n_samples
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.cfg_img = cfg_img
        self.frame_stride = frame_stride
        self.text_input = text_input
        self.multiple_cond_cfg = multiple_cond_cfg
        self.loop = loop
        self.interp = interp
        self.timestep_spacing = timestep_spacing
        self.guidance_rescale = guidance_rescale

    def generate_video(
        self,
        prompt: str,
        image_path: str,
    ):
        h, w = self.height // 8, self.width // 8
        n_frames = self.video_length
        noise_shape = [1, self.channels, self.video_length, h, w]

        with torch.no_grad(), torch.amp.autocast("cuda"):
            videos = load_data_images(
                image_path,
                video_size=(self.height, self.width),
                video_frames=n_frames,
            )
            if isinstance(videos, list):
                videos = torch.stack(videos, dim=0).to("cuda")
            else:
                videos = videos.unsqueeze(0).to("cuda")

            batch_samples = image_guided_synthesis(
                self.model,
                prompt,
                videos,
                noise_shape,
                self.n_samples,
                self.ddim_steps,
                self.ddim_eta,
                self.unconditional_guidance_scale,
                self.cfg_img,
                self.frame_stride,
                self.text_input,
                self.multiple_cond_cfg,
                self.loop,
                self.interp,
                self.timestep_spacing,
                self.guidance_rescale,
            )

            ### benchmark output
            batch_samples = batch_samples.detach().squeeze().cpu()
            batch_samples = torch.clamp(batch_samples.float(), -1.0, 1.0)
            batch_samples = (batch_samples + 1.0) / 2.0
            batch_samples = batch_samples.permute(1, 0, 2, 3)

        return batch_samples


def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:  # noqa: E722
            new_pl_sd = dict()
            for k, v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = dict()
        for key in state_dict["module"]:
            new_pl_sd[key[16:]] = state_dict["module"][key]
        model.load_state_dict(new_pl_sd)
    print(">>> model checkpoint loaded.")
    return model


def load_data_images(file_path, video_size=(256, 256), video_frames=16):
    transform = transforms.Compose(
        [
            transforms.Resize(min(video_size)),
            transforms.CenterCrop(video_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    image = Image.open(file_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(1)  # [c,1,h,w]
    frame_tensor = repeat(
        image_tensor, "c t h w -> c (repeat t) h w", repeat=video_frames
    )

    return frame_tensor


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, "b c t h w -> (b t) c h w")
    z = model.encode_first_stage(x)
    z = rearrange(z, "(b t) c h w -> b c t h w", b=b, t=t)
    return z


def image_guided_synthesis(
    model,
    prompts,
    videos,
    noise_shape,
    n_samples=1,
    ddim_steps=50,
    ddim_eta=1.0,
    unconditional_guidance_scale=1.0,
    cfg_img=None,
    fs=None,
    text_input=False,
    multiple_cond_cfg=False,
    loop=False,
    interp=False,
    timestep_spacing="uniform",
    guidance_rescale=0.0,
):
    kwargs = {}
    ddim_sampler = (
        DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    )
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""] * batch_size

    img = videos[:, :, 0]  # bchw
    img_emb = model.embedder(img)  ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}
    if model.model.conditioning_key == "hybrid":
        z = get_latent_z(model, videos)  # b c t h w
        if loop or interp:
            img_cat_cond = torch.zeros_like(z)
            img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
            img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
        else:
            img_cat_cond = z[:, :, :1, :, :]
            img_cat_cond = repeat(
                img_cat_cond, "b c t h w -> b c (repeat t) h w", repeat=z.shape[2]
            )
        cond["c_concat"] = [img_cat_cond]  # b c 1 h w

    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img))  ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
        if model.model.conditioning_key == "hybrid":
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb, img_emb], dim=1)]}
        if model.model.conditioning_key == "hybrid":
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):
        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:
            samples, _ = ddim_sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=batch_size,
                shape=noise_shape[1:],
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                cfg_img=cfg_img,
                mask=cond_mask,
                x0=cond_z0,
                fs=fs,
                timestep_spacing=timestep_spacing,
                guidance_rescale=guidance_rescale,
                **kwargs,
            )

        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)
