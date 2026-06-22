from typing import Union, List
import torch
import torch.nn.functional as F

import clip
from PIL import Image

from worldscore.benchmark.metrics.base_metrics import BaseMetric


class CLIPConsistencyMetric(BaseMetric):
    """
    
    Cosine similarity between the embeddings of two images extracted by CLIP.
    
    RANGE: [0, 1] higher the better
    """
    
    def __init__(self) -> None:
        super().__init__()
        model, preprocess = clip.load("ViT-B/32", device=self._device)
        
        self._model = model
        self._preprocess = preprocess        
    
    def _compute_scores(
        self, 
        rendered_images: List[Union[str, Image.Image]],
    ) -> float:

        def compute_similarity(image1, image2):
            # Load and preprocess images
            if isinstance(image1, str):
                image1 = Image.open(image1)
            image1 = self._preprocess(image1).unsqueeze(0).to(self._device)
            if isinstance(image2, str):
                image2 = Image.open(image2)
            image2 = self._preprocess(image2).unsqueeze(0).to(self._device)
            
            # Extract embeddings
            with torch.no_grad():
                image_features1 = self._model.encode_image(image1)
                image_features2 = self._model.encode_image(image2)
            
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