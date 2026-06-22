import numpy as np

from worldscore.benchmark.metrics.base_metrics import BaseMetric
from worldscore.common.optional_dependencies import handle_module_not_found_error

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["worldscore"])


class StructuralSimilarityIndexMeasureMetric(BaseMetric):
    """
    The Structural Similarity Index Measure (SSIM) is a method for predicting the
    perceived quality of digital television and cinematic pictures, as well as other
    kinds of digital images and videos. It is also used for measuring the similarity
    between two images. The SSIM index is a full reference metric; in other words, the
    measurement or prediction of image quality is based on an initial uncompressed or
    distortion-free image as reference.

    We use the TorchMetrics implementation:
    https://torchmetrics.readthedocs.io/en/stable/image/structural_similarity.html
    """

    def __init__(self) -> None:
        super().__init__()
        self._metric = StructuralSimilarityIndexMeasure().to(self._device)

    def _compute_scores(
        self,
        rendered_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> float:
        img1, img2 = self._process_np_to_tensor(rendered_image, reference_image)

        score: float = self._metric(img1, img2).detach().item()
        return score
