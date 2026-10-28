"""PyTorch Native MobileNetV2 ImageNet Classifier for VisionPulse."""

import time
from typing import List, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

from visionpulse.models.base import (
    AbstractVisionModel,
    ClassificationPrediction,
    InferenceResult,
)


class MobileNetClassifier(AbstractVisionModel):
    """High-performance MobileNetV2 classification engine using native PyTorch."""

    def __init__(
        self,
        device: str = "auto",
        top_k: int = 5,
        weights: Optional[MobileNet_V2_Weights] = MobileNet_V2_Weights.DEFAULT,
    ) -> None:
        super().__init__(device=device)
        self.top_k = top_k
        self.weights = weights
        self.model: Optional[torch.nn.Module] = None
        self.categories: List[str] = []
        self._transforms = None

    def load_model(self) -> None:
        """Initialize MobileNetV2 and load pretrained ImageNet-1k weights."""
        if self._is_loaded:
            return

        self.model = models.mobilenet_v2(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()

        if self.weights is not None:
            self.categories = self.weights.meta["categories"]
            self._transforms = self.weights.transforms()
        else:
            self.categories = [f"class_{i}" for i in range(1000)]

        self._is_loaded = True

    def _preprocess(self, bgr_image: np.ndarray) -> torch.Tensor:
        """Convert BGR OpenCV image to normalized PyTorch tensor."""
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Input image is empty or None")

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (224, 224), interpolation=cv2.INTER_LINEAR)

        # HWC -> CHW, float32, normalize to [0, 1]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        # ImageNet standardization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # Add batch dim and send to device
        return tensor.unsqueeze(0).to(self.device)

    def predict(
        self,
        image: np.ndarray,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> InferenceResult:
        """Run classification on input image and return Top-K predictions."""
        if not self._is_loaded:
            self.load_model()

        k = top_k or self.top_k
        start_time = time.perf_counter()

        tensor = self._preprocess(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)
            scores, indices = torch.topk(probabilities, k=min(k, len(self.categories)))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        scores_list = scores[0].cpu().tolist()
        indices_list = indices[0].cpu().tolist()

        predictions: List[ClassificationPrediction] = []
        for idx, score in zip(indices_list, scores_list):
            label = self.categories[idx] if idx < len(self.categories) else f"class_{idx}"
            predictions.append(
                ClassificationPrediction(
                    class_id=idx,
                    class_name=label.replace("_", " ").title(),
                    confidence=float(score),
                )
            )

        context = None
        if predictions:
            top = predictions[0]
            context = f"Scene classified as '{top.class_name}' with {top.confidence:.1%} confidence."

        return InferenceResult(
            model_name="MobileNetV2 (ImageNet-1k)",
            mode="classification",
            device=str(self.device),
            latency_ms=elapsed_ms,
            classifications=predictions,
            context_summary=context,
        )
