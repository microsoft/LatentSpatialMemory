from typing import List, Union

from PIL import Image

from worldscore.benchmark.metrics.base_metrics import IQAPytorchMetric


class CLIPScoreMetric(IQAPytorchMetric):
    """
    CLIPScore for no reference image caption matching.

    Reference:
        @inproceedings{hessel2021clipscore,
        title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
        author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le
            and Choi, Yejin},
        booktitle={EMNLP},
        year={2021}
        }

    Reference url: https://github.com/jmhessel/clipscore
    Re-implmented by: Chaofeng Chen (https://github.com/chaofengc)

    We use the IQA-Pytorch implementation:
    https://iqa-pytorch.readthedocs.io/

    RANGE: [0, 2.5] higher the better
    CLIP-S(c, v) = w * max(cos(c, v), 0), w=2.5
    """

    def __init__(self) -> None:
        super().__init__(metric_name="clipscore")

    def _compute_scores(
        self,
        rendered_images: List[Union[str, Image.Image]],
        caption_list: List[str],
    ) -> float:
        imgs = self._process_image(rendered_images)

        scores = []
        for img in imgs:
            score: float = self._metric(
                img.unsqueeze(0), caption_list=caption_list
            ).item()
            scores.append(score)

        score = sum(scores) / len(scores)
        return score
