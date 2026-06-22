import numpy as np

from worldscore.benchmark.metrics.base_metrics import BaseMetric
from worldscore.common.optional_dependencies import handle_module_not_found_error

try:
    from torchmetrics.image import PeakSignalNoiseRatio
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["worldscore"])


class PeakSignalNoiseRatioMetric(BaseMetric):
    """
    Peak signal-to-noise ratio (PSNR) is an engineering term for the ratio between
    the maximum possible power of a signal and the power of corrupting noise that
    affects the fidelity of its representation. Because many signals have a very wide
    dynamic range, PSNR is usually expressed as a logarithmic quantity using the decibel
    scale.

    PSNR is commonly used to quantify reconstruction quality for images and video
    subject to lossy compression.

    We use the TorchMetrics implementation:
    https://torchmetrics.readthedocs.io/en/stable/image/peak_signal_noise_ratio.html
    """

    def __init__(self) -> None:
        super().__init__()
        self._metric = PeakSignalNoiseRatio().to(self._device)

    def _compute_scores(
        self,
        rendered_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> float:
        img1, img2 = self._process_np_to_tensor(rendered_image, reference_image)

        score: float = self._metric(img1, img2).detach().item()
        return score
