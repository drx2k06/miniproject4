"""Unit tests for FastAPI REST microservice."""

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest
from visionpulse.web.api import app

client = TestClient(app)


@pytest.fixture(scope="module")
def encoded_test_image():
    """Create JPEG-encoded bytes of a test image."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 50, (0, 255, 0), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "accelerator" in data


def test_models_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 3


def test_classify_endpoint(encoded_test_image):
    response = client.post(
        "/api/v1/classify?top_k=3",
        files={"file": ("test.jpg", encoded_test_image, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "classification"
    assert len(data["classifications"]) == 3


def test_detect_endpoint(encoded_test_image):
    response = client.post(
        "/api/v1/detect?conf_threshold=0.2",
        files={"file": ("test.jpg", encoded_test_image, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "detection"
    assert "boxes" in data


def test_analyze_endpoint(encoded_test_image):
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("test.jpg", encoded_test_image, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "fusion"
    assert "classifications" in data
    assert "boxes" in data


def test_invalid_image_upload():
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.txt", b"corrupted data not an image", "text/plain")},
    )
    assert response.status_code == 400
