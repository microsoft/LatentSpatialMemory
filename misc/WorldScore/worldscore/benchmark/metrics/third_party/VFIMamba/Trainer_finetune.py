import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from model.warplayer import warp

from worldscore.benchmark.metrics.third_party.VFIMamba.config import *

def convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k and 'attn_mask' not in k and 'HW' not in k
    }

class Model:
    def __init__(self, local_rank):
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH']
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        self.name = MODEL_CONFIG['LOGNAME']
        self.device()

        self.local = LOCAL

        # train
        if local_rank != -1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def device(self):
        self.net.to(torch.device("cuda"))

    def load_model(self, name=None, rank=0, real=False):
        if rank <= 0 :
            if name is None:
                name = self.name
            print(f"loading {name} ckpt")
            self.net.load_state_dict(convert(torch.load(f'worldscore/benchmark/metrics/checkpoints/{name}.pkl')), strict=True)

    @classmethod
    def from_pretrained(cls, model_id, local_rank=-1):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "Model is hosted on the Hugging Face Hub. "
                "Please install huggingface_hub by running `pip install huggingface_hub` to load the weights correctly."
            )
        if "/" not in model_id:
            model_id = "MCG-NJU/" + model_id
        ckpt_path = hf_hub_download(repo_id=model_id, filename="model.pkl")
        print(f"loading {model_id} ckpt")
        checkpoint = torch.load(ckpt_path)
        from transformers import PretrainedConfig
        cfg = PretrainedConfig.from_pretrained(model_id)
        MODEL_CONFIG['MODEL_ARCH'] = init_model_config(
            F=cfg.F,
            depth=cfg.depth,
        )
        model = cls(local_rank)
        model.net.load_state_dict(convert(checkpoint), strict=True)
        return model

    @torch.no_grad()
    def hr_inference(self, img0, img1, local, TTA = False, down_scale = 1.0, timestep = 0.5, fast_TTA = False):
        '''
        Infer with down_scale flow
        Noting: return BxCxHxW
        '''
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)

            flow, mask = self.net.calculate_flow(imgs_down, timestep, local=local)

            flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
            mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)

            af = self.net.feature_bone(img0, img1)
            pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
            return pred

        imgs = torch.cat((img0, img1), 1)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds = infer(input)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        if TTA == False:
            return infer(imgs)
        else:
            return (infer(imgs) + infer(imgs.flip(2).flip(3)).flip(2).flip(3)) / 2

    @torch.no_grad()
    def inference(self, img0, img1, local, TTA = False, timestep = 0.5, scale=0, fast_TTA = False):
        imgs = torch.cat((img0, img1), 1)
        '''
        Noting: return BxCxHxW
        '''
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            _, _, _, preds = self.net(input, local=local, timestep=timestep, scale=scale)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        _, _, _, pred = self.net(imgs, timestep=timestep, scale=scale, local=local)
        if TTA == False:
            return pred
        else:
            _, _, _, pred2 = self.net(imgs.flip(2).flip(3), timestep=timestep, scale=scale, local=local)
            return (pred + pred2.flip(2).flip(3)) / 2
