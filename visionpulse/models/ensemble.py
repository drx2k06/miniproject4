"""Dual-Engine Fusion Engine combining MobileNetV2 and YOLOv8."""

import time
from typing import Optional
import numpy as np

from visionpulse.models.base import AbstractVisionModel, InferenceResult
from visionpulse.models.mobilenet import MobileNetClassifier
from visionpulse.models.yolo import YOLOv8Detector


class DualEngineFusion(AbstractVisionModel):
    """Ensemble perception engine performing joint scene classification and object detection."""

    def __init__(
        self,
        classifier: Optional[MobileNetClassifier] = None,
        detector: Optional[YOLOv8Detector] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self.classifier = classifier or MobileNetClassifier(device=device)
        self.detector = detector or YOLOv8Detector(device=device)

    def load_model(self) -> None:
        """Load both classification and detection models."""
        if not self.classifier.is_loaded:
            self.classifier.load_model()
        if not self.detector.is_loaded:
            self.detector.load_model()
        self._is_loaded = True

    def predict(
        self,
        image: np.ndarray,
        top_k: int = 3,
        conf_threshold: Optional[float] = None,
        **kwargs,
    ) -> InferenceResult:
        """Execute dual-engine inference and synthesize joint perception summary."""
        if not self._is_loaded:
            self.load_model()

        start_time = time.perf_counter()

        # Run both perception pathways
        clf_result = self.classifier.predict(image, top_k=top_k)
        det_result = self.detector.predict(image, conf_threshold=conf_threshold)

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Cognitive Context Synthesis
        top_scene = clf_result.classifications[0].class_name if clf_result.classifications else "Unknown"
        top_conf = clf_result.classifications[0].confidence if clf_result.classifications else 0.0

        detected_items = list(det_result.counts.keys())
        if detected_items:
            det_summary = ", ".join([f"{count} {name}" for name, count in det_result.counts.items()])
            context = (
                f"Scene context: '{top_scene}' ({top_conf:.1%}). "
                f"Localized entities: [{det_summary}]."
            )
        else:
            context = f"Scene context: '{top_scene}' ({top_conf:.1%}) with no localized foreground objects."

        return InferenceResult(
            model_name="DualEngine (MobileNetV2 + YOLOv8)",
            mode="fusion",
            device=str(self.device),
            latency_ms=total_latency_ms,
            classifications=clf_result.classifications,
            boxes=det_result.boxes,
            counts=det_result.counts,
            context_summary=context,
            extra_metadata={
                "classification_latency_ms": clf_result.latency_ms,
                "detection_latency_ms": det_result.latency_ms,
            },
        )
