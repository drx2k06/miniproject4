"""Unit tests for HUDOverlay graphics engine."""

import numpy as np
from visionpulse.core.overlay import HUDOverlay
from visionpulse.models.base import (
    BoundingBox,
    ClassificationPrediction,
    InferenceResult,
    TelemetryData,
)


def test_overlay_rendering():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    boxes = [
        BoundingBox(
            x1=50,
            y1=50,
            x2=200,
            y2=250,
            confidence=0.92,
            class_id=0,
            class_name="person",
        )
    ]

    preds = [
        ClassificationPrediction(class_id=1, class_name="Coffee Mug", confidence=0.85),
        ClassificationPrediction(class_id=2, class_name="Teapot", confidence=0.10),
    ]

    result = InferenceResult(
        model_name="TestEngine",
        mode="fusion",
        latency_ms=18.5,
        boxes=boxes,
        classifications=preds,
        context_summary="Test context summary string",
    )

    telemetry = TelemetryData(
        fps=30.0,
        latency_ms=18.5,
        resolution=(640, 480),
    )

    rendered = HUDOverlay.render_hud(frame, result, telemetry)
    assert rendered.shape == frame.shape
    # Frame should have non-zero pixels from HUD graphics
    assert np.any(rendered > 0)


def test_color_determinism():
    c1 = HUDOverlay.get_class_color("person")
    c2 = HUDOverlay.get_class_color("person")
    assert c1 == c2
    assert isinstance(c1, tuple) and len(c1) == 3
