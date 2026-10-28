"""Unit tests for StreamSource multi-source ingestion."""

import os
import cv2
import numpy as np
import pytest
from visionpulse.core.stream import StreamSource


def test_synthetic_stream():
    with StreamSource(source="synthetic", synthetic_fps=60, synthetic_res=(320, 240)) as stream:
        assert stream.is_opened
        assert stream.source_type == "synthetic"
        assert stream.resolution == (320, 240)

        ok, frame = stream.read()
        assert ok
        assert frame is not None
        assert frame.shape == (240, 320, 3)


def test_image_stream(tmp_path):
    img_path = str(tmp_path / "test_img.jpg")
    dummy = np.full((100, 200, 3), 128, dtype=np.uint8)
    cv2.imwrite(img_path, dummy)

    with StreamSource(source=img_path) as stream:
        assert stream.is_opened
        assert stream.source_type == "image"
        assert stream.resolution == (200, 100)

        ok, frame = stream.read()
        assert ok
        assert frame.shape == (100, 200, 3)


def test_nonexistent_stream():
    with pytest.raises(FileNotFoundError):
        StreamSource(source="invalid_path_to_video.mp4")
