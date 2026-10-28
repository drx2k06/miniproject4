"""Unit tests for MobileNetV2, YOLOv8, and Dual-Engine Fusion models."""

import numpy as np
import pytest
from visionpulse.models.base import BoundingBox, ClassificationPrediction, InferenceResult
from visionpulse.models.ensemble import DualEngineFusion
from visionpulse.models.mobilenet import MobileNetClassifier
from visionpulse.models.yolo import YOLOv8Detector


@pytest.fixture(scope="module")
def sample_frame():
    """Create a dummy BGR test image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple shapes
    img[100:300, 100:300] = (255, 100, 50)
    return img


def test_bounding_box_geometry():
    box = BoundingBox(
        x1=10.0,
        y1=20.0,
        x2=110.0,
        y2=120.0,
        confidence=0.88,
        class_id=0,
        class_name="person",
    )
    assert box.width == 100.0
    assert box.height == 100.0
    assert box.area == 10000.0


def test_mobilenet_classifier(sample_frame):
    clf = MobileNetClassifier(top_k=3, device="cpu")
    clf.load_model()
    assert clf.is_loaded

    result = clf.predict(sample_frame)
    assert result.mode == "classification"
    assert len(result.classifications) == 3
    assert result.latency_ms > 0
    assert 0.0 <= result.classifications[0].confidence <= 1.0
    assert result.context_summary is not None


def test_mobilenet_invalid_input():
    clf = MobileNetClassifier()
    with pytest.raises(ValueError):
        clf.predict(np.zeros((0, 0, 3), dtype=np.uint8))


def test_yolo_detector(sample_frame):
    det = YOLOv8Detector(device="cpu")
    det.load_model()
    assert det.is_loaded

    result = det.predict(sample_frame, conf_threshold=0.25)
    assert result.mode == "detection"
    assert isinstance(result.boxes, list)
    assert isinstance(result.counts, dict)
    assert result.latency_ms > 0


def test_yolo_invalid_input():
    det = YOLOv8Detector()
    with pytest.raises(ValueError):
        det.predict(None)


def test_dual_engine_fusion(sample_frame):
    fusion = DualEngineFusion(device="cpu")
    fusion.load_model()
    assert fusion.is_loaded

    result = fusion.predict(sample_frame, top_k=2)
    assert result.mode == "fusion"
    assert len(result.classifications) == 2
    assert isinstance(result.boxes, list)
    assert result.context_summary is not None
    assert "classification_latency_ms" in result.extra_metadata
    assert "detection_latency_ms" in result.extra_metadata
