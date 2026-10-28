"""Unit tests for VisionPipeline execution."""

import os
import numpy as np
import pytest
from visionpulse.config import VisionConfig
from visionpulse.core.pipeline import VisionPipeline
from visionpulse.core.stream import StreamSource


@pytest.fixture
def pipeline(tmp_path):
    cfg = VisionConfig()
    cfg.export.export_dir = str(tmp_path / "exports")
    return VisionPipeline(config=cfg, mode="detection")


def test_pipeline_modes(pipeline):
    img = np.zeros((240, 320, 3), dtype=np.uint8)

    # Detection mode
    pipeline.set_mode("detection")
    frame, res_det, tel = pipeline.process_frame(img)
    assert res_det.mode == "detection"
    assert tel.fps >= 0

    # Classification mode
    pipeline.set_mode("classification")
    frame, res_clf, tel = pipeline.process_frame(img)
    assert res_clf.mode == "classification"
    assert len(res_clf.classifications) > 0

    # Fusion mode
    pipeline.set_mode("fusion")
    frame, res_fus, tel = pipeline.process_frame(img)
    assert res_fus.mode == "fusion"


def test_invalid_mode(pipeline):
    with pytest.raises(ValueError):
        pipeline.set_mode("invalid_mode")


def test_pipeline_snapshot_and_telemetry(pipeline, tmp_path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, res, tel = pipeline.process_frame(img)

    img_path = pipeline.exporter.save_image(img, "test_snapshot.jpg")
    json_path = pipeline.exporter.save_telemetry_log(res, tel, "test_log.json")

    assert os.path.exists(img_path)
    assert os.path.exists(json_path)


def test_headless_run_live(pipeline):
    stream = StreamSource(source="synthetic", synthetic_fps=100, synthetic_res=(160, 120))
    frames_processed = []

    def on_frame(f, r, t):
        frames_processed.append(t.frame_idx)

    pipeline.run_live(stream, display=False, max_frames=3, on_frame_callback=on_frame)
    assert len(frames_processed) == 3
