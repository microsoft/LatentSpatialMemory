from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
import pyiqa
import pyiqa.models
import pyiqa.models.inference_model
import torch
from PIL import Image
from pyiqa.models.inference_model import InferenceModel
from torchvision import transforms
from torchvision.transforms import ToTensor

from worldscore.common.gpu_utils import get_torch_device
from worldscore.common.images_utils import open_image


class BaseMetric(ABC):
    """BaseMetric Class."""

    def __init__(self) -> None:
        self._metric = None
        self._device = get_torch_device()

    def _process_image(
        self,
        rendered_images: List[Union[str, Image.Image]],
    ) -> float:
        preprocessing = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        rendered_images_: List[torch.Tensor] = []
        for image in rendered_images:
            # Handle the rendered image input
            if isinstance(image, str):
                image = preprocessing(open_image(image))
            else:
                image = preprocessing(image)
            rendered_images_.append(image)

        img: torch.Tensor = torch.stack(rendered_images_).to(self._device)

        return img

    def _process_images(
        self,
        rendered_images: List[Union[str, Image.Image]],
        reference_image: Union[str, Image.Image],
    ) -> float:
        preprocessing = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        # Handle the reference image input
        if isinstance(reference_image, str):
            reference_image = preprocessing(open_image(reference_image))

        rendered_images_: List[torch.Tensor] = []
        reference_images_: List[torch.Tensor] = []
        for image in rendered_images:
            # Handle the rendered image input
            if isinstance(image, str):
                image = preprocessing(open_image(image))
            else:
                image = preprocessing(image)
            rendered_images_.append(image)

            reference_images_.append(reference_image)

        img1: torch.Tensor = torch.stack(rendered_images_).to(self._device)
        img2: torch.Tensor = torch.stack(reference_images_).to(self._device)

        return img1, img2

    def _process_np_to_tensor(
        self,
        rendered_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img1 = ToTensor()(rendered_image).unsqueeze(0).to(self._device)
        img2 = ToTensor()(reference_image).unsqueeze(0).to(self._device)
        return img1, img2

    @abstractmethod
    def _compute_scores(self, *args):
        pass


class IQAPytorchMetric(BaseMetric):
    def __init__(self, metric_name: str) -> None:
        super().__init__()
        self._metric = self._create_metric(metric_name).to(self._device)

    def _create_metric(self, metric: str) -> InferenceModel:
        metric = pyiqa.create_metric(metric)
        return metric
