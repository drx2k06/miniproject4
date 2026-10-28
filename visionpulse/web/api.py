"""Production FastAPI REST Microservice for VisionPulse."""

import io
from typing import Any, Dict
import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import psutil
import torch

from visionpulse import __author__, __version__
from visionpulse.models.ensemble import DualEngineFusion
from visionpulse.models.mobilenet import MobileNetClassifier
from visionpulse.models.yolo import YOLOv8Detector

app = FastAPI(
    title="VisionPulse Cognitive Vision API",
    description="High-throughput real-time computer vision perception microservice.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model singletons (lazy loaded)
_classifier: MobileNetClassifier = None
_detector: YOLOv8Detector = None
_fusion: DualEngineFusion = None


def get_classifier() -> MobileNetClassifier:
    global _classifier
    if _classifier is None:
        _classifier = MobileNetClassifier()
        _classifier.load_model()
    return _classifier


def get_detector() -> YOLOv8Detector:
    global _detector
    if _detector is None:
        _detector = YOLOv8Detector()
        _detector.load_model()
    return _detector


def get_fusion() -> DualEngineFusion:
    global _fusion
    if _fusion is None:
        _fusion = DualEngineFusion(classifier=get_classifier(), detector=get_detector())
        _fusion.load_model()
    return _fusion


def decode_image_upload(file_bytes: bytes) -> np.ndarray:
    """Safely decode image upload bytes to OpenCV BGR numpy array."""
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Supported formats: JPEG, PNG, WebP, BMP.",
        )
    return image


@app.get("/api/v1/health")
def health_check() -> Dict[str, Any]:
    """Return health status, hardware capabilities, and memory telemetry."""
    mem = psutil.virtual_memory()
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Multi-threaded)"

    return {
        "status": "healthy",
        "service": "VisionPulse API",
        "version": __version__,
        "author": __author__,
        "accelerator": {
            "cuda_available": cuda_available,
            "device_name": device_name,
            "torch_version": torch.__version__,
        },
        "system_memory": {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "percent_used": mem.percent,
        },
    }


@app.get("/api/v1/models")
def list_models() -> Dict[str, Any]:
    """List available model engines and specifications."""
    return {
        "models": [
            {
                "id": "mobilenet_v2",
                "name": "MobileNetV2 (ImageNet-1k)",
                "type": "Classification",
                "classes": 1000,
                "input_resolution": "224x224",
            },
            {
                "id": "yolov8n",
                "name": "YOLOv8 Nano (COCO)",
                "type": "Object Detection",
                "classes": 80,
                "input_resolution": "Dynamic",
            },
            {
                "id": "dual_engine_fusion",
                "name": "Dual-Engine Fusion",
                "type": "Ensemble Perception",
                "capabilities": ["Classification", "Spatial Detection", "Context Synthesis"],
            },
        ]
    }


@app.post("/api/v1/classify")
async def classify_image(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20),
) -> Dict[str, Any]:
    """Run MobileNetV2 top-K classification on uploaded image."""
    contents = await file.read()
    image = decode_image_upload(contents)
    model = get_classifier()
    result = model.predict(image, top_k=top_k)
    return result.model_dump()


@app.post("/api/v1/detect")
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Query(default=0.35, ge=0.0, le=1.0),
    iou_threshold: float = Query(default=0.45, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    """Run YOLOv8 object detection on uploaded image."""
    contents = await file.read()
    image = decode_image_upload(contents)
    model = get_detector()
    result = model.predict(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    return result.model_dump()


@app.post("/api/v1/analyze")
async def analyze_dual_engine(
    file: UploadFile = File(...),
    conf_threshold: float = Query(default=0.35, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    """Run joint dual-engine perception (classification + detection + cognitive context)."""
    contents = await file.read()
    image = decode_image_upload(contents)
    model = get_fusion()
    result = model.predict(image, conf_threshold=conf_threshold)
    return result.model_dump()
