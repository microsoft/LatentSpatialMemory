from typing import List, Union

from PIL import Image

from worldscore.benchmark.metrics.base_metrics import IQAPytorchMetric


class CLIPAestheticScoreMetric(IQAPytorchMetric):
    """
    We use the IQA-Pytorch implementation:
    https://iqa-pytorch.readthedocs.io/

    RANGE: [0, 10] higher the better
    """

    def __init__(self) -> None:
        super().__init__(metric_name="laion_aes")

    def _compute_scores(self, rendered_images: List[Union[str, Image.Image]]) -> float:
        imgs = self._process_image(rendered_images)

        scores = []
        for img in imgs:
            score: float = self._metric(img.unsqueeze(0)).item()
            scores.append(score)

        score = sum(scores) / len(scores)
        return score
