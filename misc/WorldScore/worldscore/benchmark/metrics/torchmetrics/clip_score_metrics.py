from typing import List, Union

import torch
from PIL import Image

from worldscore.benchmark.metrics.base_metrics import BaseMetric
from worldscore.common.optional_dependencies import handle_module_not_found_error

try:
    from torchmetrics.multimodal.clip_score import CLIPScore
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["worldscore"])


class CLIPScoreMetric(BaseMetric):
    """
    CLIP Score is a reference free metric that can be used to evaluate the correlation
    between a generated caption for an image and the actual content of the image.
    It has been found to be highly correlated with human judgement.

    We use the TorchMetrics implementation:
    https://torchmetrics.readthedocs.io/en/stable/multimodal/clip_score.html

    Range: [0, 100] higher the better
    CLIPScore(I, C) = max(100*cos(EI, EC), 0)
    """

    def __init__(self) -> None:
        super().__init__()
        self._metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(
            self._device
        )

    def _compute_scores(
        self, rendered_images: List[Union[str, Image.Image]], text_prompt: str
    ) -> float:
        imgs = self._process_image(rendered_images)

        imgs = (imgs * 255).to(torch.uint8)

        scores = []
        for img in imgs:
            score: float = self._metric(img, text_prompt).detach().item()
            scores.append(score)

        score = sum(scores) / len(scores)
        return score
