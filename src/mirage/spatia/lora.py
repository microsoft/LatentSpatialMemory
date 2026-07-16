import torch
from diffsynth.utils.lora import GeneralLoRALoader


class GeneralLoRALoaderWithUnload(GeneralLoRALoader):
    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        self.fuse_lora_to_base_model(model, state_dict_lora, alpha=alpha)

    def unload(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        updated_num = 0
        lora_name_dict = self.convert_state_dict(state_dict_lora)
        for name, module in model.named_modules():
            up_key = f"{name}.lora_B.weight"
            down_key = f"{name}.lora_A.weight"
            if up_key not in lora_name_dict or down_key not in lora_name_dict:
                continue
            weight_up = lora_name_dict[up_key].to(
                device=self.device, dtype=self.torch_dtype
            )
            weight_down = lora_name_dict[down_key].to(
                device=self.device, dtype=self.torch_dtype
            )
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2)
                weight_down = weight_down.squeeze(3).squeeze(2)
                weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(
                    2
                ).unsqueeze(3)
            else:
                weight_lora = alpha * torch.mm(weight_up, weight_down)
            state_dict = module.state_dict()
            state_dict["weight"] = (
                state_dict["weight"].to(device=self.device, dtype=self.torch_dtype)
                - weight_lora
            )
            module.load_state_dict(state_dict)
            updated_num += 1
        print(f"{updated_num} tensors are reverted by LoRA.")
