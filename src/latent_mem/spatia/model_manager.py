import importlib
import logging
import math
import os

import torch
import torch.nn as nn
from diffsynth.utils.lora import GeneralLoRALoader

from .model_config import model_loader_configs
from .utils import (
    hash_state_dict_keys,
    init_weights_on_device,
    load_state_dict,
)

logger = logging.getLogger(__name__)


def _get_module_by_name(root_module, module_path):
    if module_path == "":
        return root_module
    module = root_module
    for attr in module_path.split("."):
        if not hasattr(module, attr):
            return None
        module = getattr(module, attr)
    return module


def _materialize_and_initialize_missing_parameters(
    model, missing_keys, device, torch_dtype
):
    if not missing_keys:
        return
    for key in missing_keys:
        parent_path, _, leaf_name = key.rpartition(".")
        module = _get_module_by_name(model, parent_path)
        if module is None:
            continue

        # Handle parameters
        if hasattr(module, "_parameters") and leaf_name in module._parameters:
            param = module._parameters[leaf_name]
            if param is None:
                continue
            if getattr(param, "is_meta", False):
                new_param = torch.nn.Parameter(
                    torch.empty_like(param, device=device, dtype=torch_dtype),
                    requires_grad=param.requires_grad,
                )
                with torch.no_grad():
                    if "bias" in leaf_name:
                        torch.nn.init.zeros_(new_param)
                    elif new_param.ndim == 1:
                        torch.nn.init.ones_(new_param)
                    else:
                        torch.nn.init.kaiming_uniform_(new_param, a=math.sqrt(5))
                module._parameters[leaf_name] = new_param
            continue

        # Handle buffers (rarely meta in our flow, but keep for completeness)
        if hasattr(module, "_buffers") and leaf_name in module._buffers:
            buffer = module._buffers[leaf_name]
            if buffer is None:
                continue
            if getattr(buffer, "is_meta", False):
                module._buffers[leaf_name] = torch.empty_like(
                    buffer, device=device, dtype=torch_dtype
                )


def load_model_from_single_file_customized(
    state_dict,
    model_names,
    model_classes,
    model_resource,
    torch_dtype,
    device,
    strict_load=True,
    class_overwrite_kwargs={},
):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        logger.info(f"    model_name: {model_name} model_class: {model_class.__name__}")
        state_dict_converter = model_class.state_dict_converter()
        if model_resource == "civitai":
            state_dict_results = state_dict_converter.from_civitai(state_dict)
        elif model_resource == "diffusers":
            state_dict_results = state_dict_converter.from_diffusers(state_dict)
        if isinstance(state_dict_results, tuple):
            model_state_dict, extra_kwargs = state_dict_results
            logger.info(
                f"        This model is initialized with extra kwargs: {extra_kwargs}"
            )
        else:
            model_state_dict, extra_kwargs = state_dict_results, {}
        torch_dtype = (
            torch.float32
            if extra_kwargs.get("upcast_to_float32", False)
            else torch_dtype
        )
        extra_kwargs.update(class_overwrite_kwargs)
        with init_weights_on_device():
            model = model_class(**extra_kwargs)
        if hasattr(model, "eval"):
            model = model.eval()

        incompat = model.load_state_dict(
            model_state_dict, strict=strict_load, assign=True
        )

        if getattr(incompat, "missing_keys", None):
            logger.warning(f"Missing keys ({len(incompat.missing_keys)}): ")
            for key in incompat.missing_keys:
                logger.warning(f"    {key}")
        if getattr(incompat, "unexpected_keys", None):
            logger.warning(f"Unexpected keys ({len(incompat.unexpected_keys)}): ")
            for key in incompat.unexpected_keys:
                logger.warning(f"    {key}")
        # Ensure any remaining meta parameters/buffers from missing keys are materialized
        _materialize_and_initialize_missing_parameters(
            model, getattr(incompat, "missing_keys", []), device, torch_dtype
        )
        model = model.to(dtype=torch_dtype, device=device)
        loaded_model_names.append(model_name)
        loaded_models.append(model)
    return loaded_model_names, loaded_models


def load_model_from_single_file_slow(
    state_dict,
    model_names,
    model_classes,
    model_resource,
    torch_dtype,
    device,
    strict_load=True,
    class_overwrite_kwargs={},
):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        logger.info(f"    model_name: {model_name} model_class: {model_class.__name__}")
        state_dict_converter = model_class.state_dict_converter()
        if model_resource == "civitai":
            state_dict_results = state_dict_converter.from_civitai(state_dict)
        elif model_resource == "diffusers":
            state_dict_results = state_dict_converter.from_diffusers(state_dict)
        if isinstance(state_dict_results, tuple):
            model_state_dict, extra_kwargs = state_dict_results
            logger.info(
                f"        This model is initialized with extra kwargs: {extra_kwargs}"
            )
        else:
            model_state_dict, extra_kwargs = state_dict_results, {}
        torch_dtype = (
            torch.float32
            if extra_kwargs.get("upcast_to_float32", False)
            else torch_dtype
        )
        extra_kwargs.update(class_overwrite_kwargs)

        model = model_class(**extra_kwargs)
        if hasattr(model, "eval"):
            model = model.eval()

        incompat = model.load_state_dict(model_state_dict, strict=strict_load)

        if getattr(incompat, "missing_keys", None):
            logger.warning(f"Missing keys ({len(incompat.missing_keys)}): ")
            for key in incompat.missing_keys:
                logger.warning(f"    {key}")
        if getattr(incompat, "unexpected_keys", None):
            logger.warning(f"Unexpected keys ({len(incompat.unexpected_keys)}): ")
            for key in incompat.unexpected_keys:
                logger.warning(f"    {key}")
        model = model.to(dtype=torch_dtype, device=device)
        loaded_model_names.append(model_name)
        loaded_models.append(model)
    return loaded_model_names, loaded_models


class ModelDetectorFromSingleFile:
    def __init__(self, model_loader_configs=[], strict_load=True):
        self.strict_load = strict_load
        self.keys_hash_with_shape_dict = {}
        self.keys_hash_dict = {}
        for metadata in model_loader_configs:
            self.add_model_metadata(*metadata)

    def add_model_metadata(
        self,
        keys_hash,
        keys_hash_with_shape,
        model_names,
        model_classes,
        model_resource,
    ):
        self.keys_hash_with_shape_dict[keys_hash_with_shape] = (
            model_names,
            model_classes,
            model_resource,
        )
        if keys_hash is not None:
            self.keys_hash_dict[keys_hash] = (
                model_names,
                model_classes,
                model_resource,
            )

    def match(self, file_path="", state_dict={}):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            return True
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            return True
        return False

    def load(
        self,
        file_path="",
        state_dict={},
        device="cuda",
        torch_dtype=torch.float16,
        class_overwrite_kwargs={},
        **kwargs,
    ):
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)

        # Load models with strict matching
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            model_names, model_classes, model_resource = self.keys_hash_with_shape_dict[
                keys_hash_with_shape
            ]
            loaded_model_names, loaded_models = load_model_from_single_file_customized(
                state_dict,
                model_names,
                model_classes,
                model_resource,
                torch_dtype,
                device,
                self.strict_load,
                class_overwrite_kwargs,
            )
            return loaded_model_names, loaded_models

        # Load models without strict matching
        # (the shape of parameters may be inconsistent, and the state_dict_converter will modify the model architecture)
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            model_names, model_classes, model_resource = self.keys_hash_dict[keys_hash]
            loaded_model_names, loaded_models = load_model_from_single_file_customized(
                state_dict,
                model_names,
                model_classes,
                model_resource,
                torch_dtype,
                device,
                self.strict_load,
                class_overwrite_kwargs,
            )
            return loaded_model_names, loaded_models

        return loaded_model_names, loaded_models


class ModelManager:
    def __init__(
        self,
        torch_dtype=torch.float16,
        device="cuda",
        model_id_list: list[str] | None = None,
        downloading_priority: list[str] | None = None,
        file_path_list: list[str] | None = None,
        strict_load=True,
    ):
        if model_id_list:
            raise ValueError(
                "DiffSynth 2.x compatibility only supports explicit local model "
                "files in latent_mem.spatia.ModelManager."
            )
        self.torch_dtype = torch_dtype
        self.device = device
        self.model = []
        self.model_path = []
        self.model_name = []
        self.model_detector = [
            ModelDetectorFromSingleFile(model_loader_configs, strict_load=strict_load),
        ]
        self.load_models(file_path_list or [])

    def load_model_from_single_file(
        self,
        file_path="",
        state_dict={},
        model_names=[],
        model_classes=[],
        model_resource=None,
    ):
        print(f"Loading models from file: {file_path}")
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        model_names, models = load_model_from_single_file_customized(
            state_dict,
            model_names,
            model_classes,
            model_resource,
            self.torch_dtype,
            self.device,
        )
        for model_name, model in zip(model_names, models):
            self.model.append(model)
            self.model_path.append(file_path)
            self.model_name.append(model_name)
        print(f"    The following models are loaded: {model_names}.")

    def load_lora(self, file_path="", state_dict={}, lora_alpha=1.0):
        if isinstance(file_path, list):
            for file_path_ in file_path:
                self.load_lora(file_path_, state_dict=state_dict, lora_alpha=lora_alpha)
        else:
            print(f"Loading LoRA models from file: {file_path}")
            is_loaded = False
            if len(state_dict) == 0:
                state_dict = load_state_dict(file_path)
            for model_name, model, model_path in zip(
                self.model_name, self.model, self.model_path
            ):
                loader = GeneralLoRALoader(
                    torch_dtype=self.torch_dtype,
                    device=self.device,
                )
                lora_layers = loader.get_name_dict(state_dict)
                target_layers = {name for name, _ in model.named_modules()}
                if not target_layers.intersection(lora_layers):
                    continue
                print(f"    Adding LoRA to {model_name} ({model_path}).")
                loader.fuse_lora_to_base_model(model, state_dict, alpha=lora_alpha)
                is_loaded = True
            if not is_loaded:
                print(f"    Cannot load LoRA: {file_path}")

    def load_model(
        self,
        file_path,
        model_names=None,
        device=None,
        torch_dtype=None,
        class_overwrite_kwargs={},
    ):
        print(f"Loading models from: {file_path}")
        if device is None:
            device = self.device
        if torch_dtype is None:
            torch_dtype = self.torch_dtype
        if isinstance(file_path, list):
            state_dict = {}
            for path in file_path:
                state_dict.update(load_state_dict(path))
        elif os.path.isfile(file_path):
            state_dict = load_state_dict(file_path)
        else:
            state_dict = None
        for model_detector in self.model_detector:
            if model_detector.match(file_path, state_dict):
                model_names, models = model_detector.load(
                    file_path,
                    state_dict,
                    device=device,
                    torch_dtype=torch_dtype,
                    allowed_model_names=model_names,
                    model_manager=self,
                    class_overwrite_kwargs=class_overwrite_kwargs,
                )
                for model_name, model in zip(model_names, models):
                    self.model.append(model)
                    self.model_path.append(file_path)
                    self.model_name.append(model_name)
                print(f"    The following models are loaded: {model_names}.")
                break
        else:
            print(f"    We cannot detect the model type. No models are loaded.")

    def load_models(
        self,
        file_path_list,
        model_names=None,
        device=None,
        torch_dtype=None,
        class_overwrite_kwargs={},
    ):
        for file_path in file_path_list:
            self.load_model(
                file_path,
                model_names,
                device=device,
                torch_dtype=torch_dtype,
                class_overwrite_kwargs=class_overwrite_kwargs,
            )

    def fetch_model(
        self, model_name, file_path=None, require_model_path=False, index=None
    ):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(
            self.model, self.model_path, self.model_name
        ):
            if file_path is not None and file_path != model_path:
                continue
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            print(f"No {model_name} models available.")
            return None
        if len(fetched_models) == 1:
            print(f"Using {model_name} from {fetched_model_paths[0]}.")
            model = fetched_models[0]
            path = fetched_model_paths[0]
        else:
            if index is None:
                model = fetched_models[0]
                path = fetched_model_paths[0]
                print(
                    f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {fetched_model_paths[0]}."
                )
            elif isinstance(index, int):
                model = fetched_models[:index]
                path = fetched_model_paths[:index]
                print(
                    f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {fetched_model_paths[:index]}."
                )
            else:
                model = fetched_models
                path = fetched_model_paths
                print(
                    f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {fetched_model_paths}."
                )
        if require_model_path:
            return model, path
        else:
            return model

    def to(self, device):
        for model in self.model:
            model.to(device)
