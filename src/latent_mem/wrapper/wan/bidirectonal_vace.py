import gc
import os
import types
from dataclasses import asdict, is_dataclass

import torch

from latent_mem.configs.model_config import BackboneConfig
from latent_mem.utils.scheduler import FlowMatchScheduler, SchedulerInterface
from latent_mem.wan.modules.model_spatia import SpatiaWanModel


class BidirectionalWanWrapperVACE(torch.nn.Module):
    """
    Wrapper for Spatia: Camera-controlled video generation with VACE.

    Uses SpatiaWanModel which is optimized for:
    - Per-frame timestep (R+P frames get t=0, T frames get sampled t)
    - VACE processes P+T scene projections while main model has R+P+T frames

    VACE initialization lives in the wrapper:
    - main backbone weights come from official Wan checkpoints
    - VACE blocks are initialized from the matching backbone blocks
    - optional VACE checkpoints can override the control branch afterwards
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        # Spatia always uses per-frame timestep
        self.uniform_timestep = False

        wan_model_path = self.config.wan_model_path
        timestep_shift = self.config.timestep_shift
        use_lora = self.config.use_lora

        print(f"load_official_backbone=True, use_lora={use_lora}")

        if is_dataclass(config) and not isinstance(config, type):
            config_dict = asdict(config)
        else:
            config_dict = config

        self.model = SpatiaWanModel(**config_dict)
        self.model.eval()

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        num_latent_frames = config.image_or_video_shape[1]
        latent_h = config.image_or_video_shape[-2]
        latent_w = config.image_or_video_shape[-1]

        self.seq_len = num_latent_frames * latent_h * latent_w // 4

        if wan_model_path is not None:
            self._load_official_weights(wan_model_path)

        if self.model.enable_vace:
            self.model.initialize_vace_from_backbone()

        self.requires_grad_(False)
        torch.cuda.empty_cache()

    def _load_official_weights(self, wan_model_name: str):
        print(f"Loading official Wan weights from: {wan_model_name}")

        backbone_state_dict = self._load_filtered_state_dict(
            wan_model_name, filter_fn=lambda k: not self._is_submodule_key(k)
        )

        missing, unexpected = self.model.load_state_dict(
            backbone_state_dict, assign=True, strict=False
        )

        if missing:
            submodule_missing = [k for k in missing if self._is_submodule_key(k)]
            backbone_missing = [k for k in missing if not self._is_submodule_key(k)]
            if backbone_missing:
                print(f"WARNING: Missing backbone keys: {backbone_missing[:5]}...")
            if submodule_missing:
                print(
                    f"INFO: Submodule keys not loaded (expected): {len(submodule_missing)} keys"
                )

        assert len(unexpected) == 0, (
            f"Official weights load failed! Unexpected keys: {unexpected}"
        )

        del backbone_state_dict
        gc.collect()
        torch.cuda.empty_cache()
        print("Official weights loaded successfully")

    @staticmethod
    def _is_submodule_key(key: str) -> bool:
        submodule_prefixes = [
            "vace_blocks",
            "vace_patch_embedding",
        ]
        return any(prefix in key for prefix in submodule_prefixes)

    def _load_checkpoint_file(self, ckpt_path: str) -> dict:
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(ckpt_path, device="cpu")
        else:
            state_dict = torch.load(
                ckpt_path, map_location="cpu", weights_only=False, mmap=True
            )

        if state_dict.get("format_version") == 2:
            raise ValueError(
                "Structured v2 training checkpoints are not valid backbone weights. "
            )

        if "generator" in state_dict or "generator_ema" in state_dict:
            if "generator_ema" in state_dict and self.config.load_ema:
                raise ValueError("EMA checkpoints are no longer supported.")
            if "generator" in state_dict:
                print("Loading generator weights")
                state_dict = state_dict["generator"]
            else:
                raise ValueError("EMA checkpoints are no longer supported.")
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]

        state_dict = {
            self._normalize_checkpoint_key(key): value
            for key, value in state_dict.items()
        }

        return state_dict

    def _load_filtered_state_dict(self, model_path: str, filter_fn=None) -> dict:
        import glob
        import json

        from safetensors.torch import load_file

        if os.path.isdir(model_path):
            index_file = os.path.join(
                model_path, "diffusion_pytorch_model.safetensors.index.json"
            )
            if os.path.exists(index_file):
                print(f"Loading state_dict from sharded files: {index_file}")
                with open(index_file, "r") as f:
                    index = json.load(f)

                weight_map = index["weight_map"]
                if filter_fn is not None:
                    weight_map = {k: v for k, v in weight_map.items() if filter_fn(k)}

                needed_shards = set(weight_map.values())
                print(
                    "  Optimized loading: using "
                    f"{len(needed_shards)}/{len(set(index['weight_map'].values()))} shards"
                )

                state_dict = {}
                for shard_file in sorted(needed_shards):
                    shard_path = os.path.join(model_path, shard_file)
                    print(f"  Loading shard: {shard_file}")
                    shard_dict = load_file(shard_path, device="cpu")

                    filtered_shard = self._filter_state_dict(shard_dict, filter_fn)
                    del shard_dict
                    state_dict.update(filtered_shard)
                    del filtered_shard
                    gc.collect()

                return state_dict

            candidate_files = [
                os.path.join(model_path, "diffusion_pytorch_model.safetensors"),
                *sorted(glob.glob(os.path.join(model_path, "*.safetensors"))),
                *sorted(glob.glob(os.path.join(model_path, "*.pt"))),
                *sorted(glob.glob(os.path.join(model_path, "*.pth"))),
            ]
            for candidate_file in candidate_files:
                if os.path.exists(candidate_file):
                    raw_state_dict = self._load_checkpoint_file(candidate_file)
                    return self._filter_state_dict(raw_state_dict, filter_fn)

            raise FileNotFoundError(
                f"No model weights found in directory: {model_path}"
            )

        raw_state_dict = self._load_checkpoint_file(model_path)
        return self._filter_state_dict(raw_state_dict, filter_fn)

    @staticmethod
    def _normalize_checkpoint_key(key: str) -> str:
        while key.startswith("model."):
            key = key.removeprefix("model.")
        return key

    @staticmethod
    def _filter_state_dict(state_dict: dict, filter_fn=None) -> dict:
        normalized_state_dict = {
            BidirectionalWanWrapperVACE._normalize_checkpoint_key(key): value
            for key, value in state_dict.items()
        }
        if filter_fn is None:
            return normalized_state_dict
        return {
            key: value for key, value in normalized_state_dict.items() if filter_fn(key)
        }

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def forward(
        self,
        noisy_image_or_video,
        timestep: torch.Tensor,
        context,
        clip_fea=None,
        y=None,
        vace_context=None,
        vace_context_scale=1.0,
        vace_hint_offset=0,
        **kwargs,
    ) -> torch.Tensor:
        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        # Calculate seq_len based on actual input shape (may include R+P+T frames for Spatia)
        # noisy_image_or_video: [B, F, C, H, W]
        actual_frames = noisy_image_or_video.shape[1]
        H, W = noisy_image_or_video.shape[-2:]
        actual_seq_len = (
            actual_frames * (H // 2) * (W // 2)
        )  # after patch embedding with stride (1,2,2)

        # vace_context is already [C, T, H, W] format from task layer
        flow_pred = self.model(
            x=noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=input_timestep,
            context=context,
            clip_fea=clip_fea,
            y=list(y)
            if y is not None
            else None,  # y is [B, C, F, H, W], model expects list of [C, F, H, W]
            vace_context=vace_context,
            vace_context_scale=vace_context_scale,
            vace_hint_offset=vace_hint_offset,
            seq_len=actual_seq_len,
            **kwargs,
        ).permute(0, 2, 1, 3, 4)

        return flow_pred

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler
        )
        self.scheduler = scheduler
        return scheduler
