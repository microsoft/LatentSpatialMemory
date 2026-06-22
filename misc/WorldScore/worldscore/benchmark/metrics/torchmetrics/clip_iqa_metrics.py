from typing import List, Union

import torch
from PIL import Image

from worldscore.benchmark.metrics.base_metrics import BaseMetric
from worldscore.common.optional_dependencies import handle_module_not_found_error

try:
    from torchmetrics.multimodal import CLIPImageQualityAssessment
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["worldscore"])


class CLIPImageQualityAssessmentMetric(BaseMetric):
    """
    Calculates CLIP-IQA, that can be used to measure the visual content of images.
    The metric is based on the CLIP model, which is a neural network trained on a
    variety of (image, text) pairs to be able to generate a vector representation of the
    image and the text that is similar if the image and text are semantically similar.

    We use the TorchMetrics implementation:
    https://torchmetrics.readthedocs.io/en/stable/multimodal/clip_iqa.html
    """

    def __init__(self) -> None:
        super().__init__()
        self._metric = CLIPImageQualityAssessment(
            model_name_or_path="openai/clip-vit-base-patch16"
        ).to(self._device)

    def _compute_scores(self, rendered_images: List[Union[str, Image.Image]]) -> float:
        imgs = self._process_image(rendered_images)

        imgs = (imgs * 255).to(torch.uint8)

        scores: float = self._metric(imgs)
        score = scores.detach().mean().item()

        return score
