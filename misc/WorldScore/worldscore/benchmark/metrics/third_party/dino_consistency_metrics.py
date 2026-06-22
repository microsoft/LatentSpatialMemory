from typing import Union, List
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image

from worldscore.benchmark.metrics.base_metrics import BaseMetric


# Function to preprocess the input image
def preprocess_image(image):
    transform = T.Compose([
        T.Resize(518),  # Rescale shorter edge to 518 pixels
        T.CenterCrop(518),  # Center crop to a 518x518 image
        T.ToTensor(),  # Convert to tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

class DINOConsistencyMetric(BaseMetric):
    """
    
    Cosine similarity between the embeddings of two images extracted by DINOv2.
    
    RANGE: [0, 1] higher the better
    """
    
    def __init__(self) -> None:
        super().__init__()
        # Load the DINOv2 model from torch.hub
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')  # vitb14 is one version, you can choose another
        
        self._model = model.to(self._device)
        self._preprocess = preprocess_image        
    
    def _compute_scores(
        self, 
        rendered_images: List[Union[str, Image.Image]],
    ) -> float:

        def compute_similarity(image1, image2):
            # Load and preprocess images
            if isinstance(image1, str):
                image1 = Image.open(image1)
            image1 = self._preprocess(image1)
            image1 = image1.to(self._device)
            if isinstance(image2, str):
                image2 = Image.open(image2)
            image2 = self._preprocess(image2)
            image2 = image2.to(self._device)
            
            # Extract embeddings
            with torch.no_grad():
                image_features1 = self._model(image1)
                image_features2 = self._model(image2)
            
            # Normalize embeddings
            image_features1 /= image_features1.norm(dim=-1, keepdim=True)
            image_features2 /= image_features2.norm(dim=-1, keepdim=True)
    
            # Compute cosine similarity (CLIP consistency)
            similarity = F.cosine_similarity(image_features1, image_features2).item()
            return similarity
        
        scores = []
        stride = 10
        
        if len(rendered_images) <= stride:
            image1 = rendered_images[0]
            image2 = rendered_images[-1]
            similarity = compute_similarity(image1, image2)
            scores.append(similarity)
        else:
            for i in range(len(rendered_images) - stride):
                image1 = rendered_images[i]
                image2 = rendered_images[i + stride]
                similarity = compute_similarity(image1, image2)
                scores.append(similarity)

        score = sum(scores) / len(scores)
        return score