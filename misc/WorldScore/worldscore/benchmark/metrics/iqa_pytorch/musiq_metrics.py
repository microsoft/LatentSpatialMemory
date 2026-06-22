from typing import List, Union

from PIL import Image

from worldscore.benchmark.metrics.base_metrics import IQAPytorchMetric


class MultiScaleImageQualityMetric(IQAPytorchMetric):
    """
    MUSIQ model.

    Reference:
            Ke, Junjie, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang.
            "Musiq: Multi-scale image quality transformer."
            ICCV 2021.

    Ref url: https://github.com/google-research/google-research/tree/master/musiq
    Re-implemented by: Chaofeng Chen (https://github.com/chaofengc)


    We use the IQA-Pytorch implementation:
    https://iqa-pytorch.readthedocs.io/

    Range: [0, 100] higher the better
    """

    def __init__(self) -> None:
        super().__init__(metric_name="musiq-paq2piq")

    def _compute_scores(self, rendered_images: List[Union[str, Image.Image]]) -> float:
        imgs = self._process_image(rendered_images)

        scores = []
        for img in imgs:
            score: float = self._metric(img.unsqueeze(0)).item()
            scores.append(score)

        score = sum(scores) / len(scores)
        return score
