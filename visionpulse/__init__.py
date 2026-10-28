"""VisionPulse: Dual-Engine Real-Time Computer Vision & Cognitive Object Intelligence System.

Re-engineered and modernized by Daniel Richard Xavier.
Original concept by kirankumarnscr7-cmyk.
"""

__version__ = "2.0.0"
__author__ = "Daniel Richard Xavier"
__license__ = "MIT"

from visionpulse.config import VisionConfig
from visionpulse.models.base import (
    AbstractVisionModel,
    BoundingBox,
    ClassificationPrediction,
    InferenceResult,
    TelemetryData,
)
from visionpulse.models.mobilenet import MobileNetClassifier
from visionpulse.models.yolo import YOLOv8Detector
from visionpulse.models.ensemble import DualEngineFusion
from visionpulse.core.pipeline import VisionPipeline
from visionpulse.core.stream import StreamSource

__all__ = [
    "VisionConfig",
    "AbstractVisionModel",
    "BoundingBox",
    "ClassificationPrediction",
    "InferenceResult",
    "TelemetryData",
    "MobileNetClassifier",
    "YOLOv8Detector",
    "DualEngineFusion",
    "VisionPipeline",
    "StreamSource",
]
