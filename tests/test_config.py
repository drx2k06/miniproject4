"""Unit tests for VisionPulse configuration management."""

import pytest
from visionpulse.config import (
    ExportConfig,
    ModelConfig,
    StreamConfig,
    VisionConfig,
    get_config,
)


def test_default_config():
    cfg = get_config()
    assert cfg.app_name == "VisionPulse"
    assert cfg.env in ("development", "production", "test")
    assert cfg.models.yolo_model == "yolov8n.pt"
    assert 0.0 <= cfg.models.yolo_conf_threshold <= 1.0
    assert 1 <= cfg.models.mobilenet_top_k <= 20
    assert cfg.stream.camera_index >= 0
    assert cfg.stream.frame_width > 0
    assert cfg.stream.frame_height > 0


def test_custom_model_config():
    cfg = ModelConfig(
        device="cpu",
        yolo_model="yolov8s.pt",
        yolo_conf_threshold=0.5,
        mobilenet_top_k=3,
    )
    assert cfg.device == "cpu"
    assert cfg.yolo_model == "yolov8s.pt"
    assert cfg.yolo_conf_threshold == 0.5
    assert cfg.mobilenet_top_k == 3


def test_invalid_conf_threshold():
    with pytest.raises(Exception):
        ModelConfig(yolo_conf_threshold=1.5)  # Out of range [0, 1]


def test_export_config():
    exp = ExportConfig(export_dir="./custom_out")
    assert exp.export_dir == "./custom_out"
