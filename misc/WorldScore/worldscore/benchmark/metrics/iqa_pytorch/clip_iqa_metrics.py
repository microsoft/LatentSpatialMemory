"""
This has several CLIP-IQA Metrics that have been proposed by
    Exploring CLIP for Assessing the Look and Feel of Images.
    Jianyi Wang Kelvin C.K. Chan Chen Change Loy.
    AAAI 2023.

Ref url: https://github.com/IceClear/CLIP-IQA
Re-implmented by: Chaofeng Chen (https://github.com/chaofengc).
Modifications:
    - We assemble multiple prompts to improve the results of clipiqa model.

We use the IQA-Pytorch implementation:
https://iqa-pytorch.readthedocs.io/

prompts used
    [
        'Good image', 'bad image',
        'Sharp image', 'blurry image',
        'sharp edges', 'blurry edges',
        'High resolution image', 'low resolution image',
        'Noise-free image', 'noisy image',
    ]

RANGE: [0, 1] higher the better
"""

from typing import List, Union

from PIL import Image

from worldscore.benchmark.metrics.base_metrics import IQAPytorchMetric


class CLIPImageQualityAssessmentMetric(IQAPytorchMetric):
    """CLIP-IQA"""

    def __init__(self, metric_name: str = "clipiqa") -> None:
        super().__init__(metric_name=metric_name)

    def _compute_scores(self, rendered_images: List[Union[str, Image.Image]]) -> float:
        imgs = self._process_image(rendered_images)

        scores = []
        for img in imgs:
            score: float = self._metric(img.unsqueeze(0)).item()
            scores.append(score)

        score = sum(scores) / len(scores)
        return score


class CLIPImageQualityAssessmentPlusMetric(CLIPImageQualityAssessmentMetric):
    """CLIP-IQA+"""

    def __init__(self) -> None:
        super().__init__(metric_name="clipiqa+")


class CLIPImageQualityAssessmentPlusRN50_512Metric(CLIPImageQualityAssessmentMetric):
    """CLIP-IQA+ ResNet-50"""

    def __init__(self) -> None:
        super().__init__(metric_name="clipiqa+_rn50_512")


class CLIPimageQualityAssessmentPlusVITL14_512Metric(CLIPImageQualityAssessmentMetric):
    """CLIP-IQA+ ViT-L14"""

    def __init__(self) -> None:
        super().__init__(metric_name="clipiqa+_vitL14_512")
