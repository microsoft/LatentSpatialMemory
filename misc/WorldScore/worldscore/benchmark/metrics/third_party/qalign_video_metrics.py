from typing import Union, List
import torch
import torch.nn.functional as F

import clip
from PIL import Image
from transformers import AutoModelForCausalLM

from worldscore.benchmark.metrics.base_metrics import BaseMetric


class QAlignVideoMetric(BaseMetric):
    """
    Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels

    We use the official implementation:
    https://github.com/Q-Future/Q-Align
    """

    def __init__(self, task: str = "aesthetic") -> None:
        super().__init__()
        self._model = AutoModelForCausalLM.from_pretrained("q-future/one-align", trust_remote_code=True, 
                                             torch_dtype=torch.float16, device_map="auto")
        self.task = task

    def _compute_scores(self, rendered_images: List[Union[str, Image.Image]]) -> float:
        images = []
        for image in rendered_images:
            if isinstance(image, str):
                image = Image.open(image)
            images.append(image)
            
        score = self._model.score([images], task_=self.task, input_="video") # task_ : quality | aesthetics; # input_: image | video
        return score.item()


class QAlignVideoAestheticMetric(QAlignVideoMetric):
    """QAlign Video Aesthetic"""

    def __init__(self) -> None:
        super().__init__(task="aesthetic")


class QAlignVideoQualityMetric(QAlignVideoMetric):
    """QAlign Video Quality"""

    def __init__(self) -> None:
        super().__init__(task="quality")
