"""Ultralytics YOLOv8 Object Detection Engine for VisionPulse."""

from collections import Counter
import time
from typing import Dict, List, Optional
import numpy as np
import torch
from ultralytics import YOLO

from visionpulse.models.base import (
    AbstractVisionModel,
    BoundingBox,
    InferenceResult,
)


class YOLOv8Detector(AbstractVisionModel):
    """High-performance YOLOv8 multi-class object detection engine."""

    def __init__(
        self,
        model_name_or_path: str = "yolov8n.pt",
        device: str = "auto",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ) -> None:
        super().__init__(device=device)
        self.model_path = model_name_or_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.yolo: Optional[YOLO] = None
        self.names: Dict[int, str] = {}

    def load_model(self) -> None:
        """Load YOLO model weights and target device."""
        if self._is_loaded:
            return

        self.yolo = YOLO(self.model_path)
        # Note: Ultralytics handles device assignment during predict or via .to()
        self.names = self.yolo.names
        self._is_loaded = True

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        **kwargs,
    ) -> InferenceResult:
        """Run object detection on input frame and return structured bounding boxes."""
        if not self._is_loaded:
            self.load_model()

        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        dev_arg = str(self.device)

        start_time = time.perf_counter()

        # Run inference
        results = self.yolo.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=dev_arg,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        boxes: List[BoundingBox] = []
        class_names_detected: List[str] = []

        if results and len(results) > 0:
            det = results[0]
            boxes_data = det.boxes

            if boxes_data is not None and len(boxes_data) > 0:
                coords = boxes_data.xyxy.cpu().numpy()
                confs = boxes_data.conf.cpu().numpy()
                classes = boxes_data.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), score, cls_id in zip(coords, confs, classes):
                    name = self.names.get(cls_id, f"object_{cls_id}")
                    boxes.append(
                        BoundingBox(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            confidence=float(score),
                            class_id=int(cls_id),
                            class_name=name,
                        )
                    )
                    class_names_detected.append(name)

        counts = dict(Counter(class_names_detected))

        if counts:
            summary_parts = [f"{count} {name}{'s' if count > 1 else ''}" for name, count in counts.items()]
            context = f"Detected: {', '.join(summary_parts)}."
        else:
            context = "No objects detected above confidence threshold."

        return InferenceResult(
            model_name=f"YOLOv8 ({self.model_path})",
            mode="detection",
            device=str(self.device),
            latency_ms=elapsed_ms,
            boxes=boxes,
            counts=counts,
            context_summary=context,
        )
