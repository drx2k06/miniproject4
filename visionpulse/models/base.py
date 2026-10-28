"""Base schemas and abstract model interface for VisionPulse."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
import torch


class BoundingBox(BaseModel):
    """Normalized or pixel bounding box coordinates for a detected object."""

    x1: float = Field(description="Top-left X coordinate")
    y1: float = Field(description="Top-left Y coordinate")
    x2: float = Field(description="Bottom-right X coordinate")
    y2: float = Field(description="Bottom-right Y coordinate")
    confidence: float = Field(description="Detection confidence [0.0, 1.0]")
    class_id: int = Field(description="Integer class index")
    class_name: str = Field(description="Semantic class label")

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


class ClassificationPrediction(BaseModel):
    """Top-K classification prediction output."""

    class_id: int
    class_name: str
    confidence: float = Field(description="Softmax probability [0.0, 1.0]")


class TelemetryData(BaseModel):
    """Real-time performance and pipeline telemetry."""

    frame_idx: int = 0
    fps: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_latency_ms: float = 0.0
    resolution: tuple[int, int] = (0, 0)
    device: str = "cpu"


class InferenceResult(BaseModel):
    """Unified container for all vision model outputs."""

    model_name: str
    mode: str = "detection"  # 'detection', 'classification', 'fusion'
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    device: str = "cpu"
    latency_ms: float = 0.0

    # For classification
    classifications: List[ClassificationPrediction] = Field(default_factory=list)

    # For detection
    boxes: List[BoundingBox] = Field(default_factory=list)
    counts: Dict[str, int] = Field(default_factory=dict)

    # Fusion / context summary
    context_summary: Optional[str] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def primary_label(self) -> str:
        """Returns the most prominent label from detection or classification."""
        if self.mode == "classification" and self.classifications:
            return f"{self.classifications[0].class_name} ({self.classifications[0].confidence:.1%})"
        if self.boxes:
            best_box = max(self.boxes, key=lambda b: b.confidence)
            return f"{best_box.class_name} ({best_box.confidence:.1%})"
        return "No Detections"


class AbstractVisionModel(ABC):
    """Abstract contract for all VisionPulse perception engines."""

    def __init__(self, device: str = "auto") -> None:
        self.requested_device = device
        self.device = self._resolve_device(device)
        self._is_loaded = False

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve device string to available PyTorch device."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        elif device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load_model(self) -> None:
        """Load weights into memory on target device."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> InferenceResult:
        """Execute inference on a BGR or RGB numpy image.

        Args:
            image: Input image array (H, W, C) in BGR format (OpenCV standard).
        Returns:
            InferenceResult containing structured predictions.
        """
        pass
