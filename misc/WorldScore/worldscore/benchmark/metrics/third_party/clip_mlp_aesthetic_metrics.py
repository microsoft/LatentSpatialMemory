from typing import Union, List
import pytorch_lightning as pl
import torch.nn as nn
import torch

import clip
from PIL import Image

from worldscore.benchmark.metrics.base_metrics import BaseMetric

# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

class CLIPMLPAestheticScoreMetric(BaseMetric):
    """
    
    We use the CLIP+MLP Aesthetic Score Predictor implementation:
    https://github.com/christophschuhmann/improved-aesthetic-predictor
    
    Range: [0, 10] higher the better
    """
    
    def __init__(self) -> None:
        super().__init__()
        model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
        s = torch.load("worldscore/benchmark/metrics/checkpoints/sac+logos+ava1-l14-linearMSE.pth", map_location=torch.device('cpu'))   # load the model you trained previously or the model available in this repo

        model.load_state_dict(s)

        model.to(self._device)
        model.eval()

        model2, preprocess = clip.load("ViT-L/14", device=self._device)  #RN50x64

        self._model = model
        self._model2 = model2
        self._preprocess = preprocess        
        
    def _compute_scores(
        self, 
        rendered_images: List[Union[str, Image.Image]],
    ) -> float:
        try:
            from torchmetrics.multimodal.clip_score import CLIPScore
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["worldscore"])
    
        scores = []
            
        for i, image in enumerate(rendered_images):
            if isinstance(image, str):
                image = Image.open(image)
            image = self._preprocess(image).unsqueeze(0).to(self._device)
            with torch.no_grad():
                image_features = self._model2.encode_image(image)
            im_emb_arr = normalized(image_features.cpu().detach().numpy())
            prediction = self._model(torch.from_numpy(im_emb_arr).type(torch.FloatTensor).to(self._device))
            scores.append(prediction.detach()[0].item())

        score = sum(scores) / len(scores)
        return score