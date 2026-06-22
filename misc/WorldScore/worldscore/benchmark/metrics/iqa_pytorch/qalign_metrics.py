from typing import List, Union

from PIL import Image

from worldscore.benchmark.metrics.base_metrics import IQAPytorchMetric


class QAlignMetric(IQAPytorchMetric):
    """
    Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels

    We use the IQA-Pytorch implementation:
    https://iqa-pytorch.readthedocs.io/
    """

    def __init__(self, metric_name: str = "qalign", task: str = "aesthetic") -> None:
        super().__init__(metric_name=metric_name)
        self.task = task

    def _compute_scores(self, rendered_images: List[Union[str, Image.Image]]) -> float:
        imgs = self._process_image(rendered_images)

        scores = []
        for img in imgs:
            score: float = self._metric(img.unsqueeze(0), task_=self.task).item()
            scores.append(score)

        score = sum(scores) / len(scores)
        return score


class QAlignAestheticMetric(QAlignMetric):
    """QAlign Image Aesthetic"""

    def __init__(self) -> None:
        super().__init__(metric_name="qalign", task="aesthetic")


class QAlignQualityMetric(QAlignMetric):
    """QAlign Image Quality"""

    def __init__(self) -> None:
        super().__init__(metric_name="qalign", task="quality")
