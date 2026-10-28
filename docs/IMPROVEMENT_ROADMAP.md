# Engineering Improvement Roadmap: VisionPulse

**Project:** VisionPulse Dual-Engine Cognitive Vision Platform  
**Lead Engineer:** Daniel Richard Xavier  
**Baseline Repository:** `https://github.com/kirankumarnscr7-cmyk/mini-project4`  
**Target Repository:** `https://github.com/drx2k06/miniproject4.git`  

---

## 1. Modernization Objectives

The objective is to elevate a 49-line proof-of-concept webcam script into an enterprise-grade, high-throughput computer vision intelligence system. The roadmap is organized across five distinct phases:

```
[Phase 1: Architecture & Runtimes] ➔ [Phase 2: Ingestion & Degradation] ➔ [Phase 3: Telemetry & Overlay] ➔ [Phase 4: Web Studio & API] ➔ [Phase 5: Benchmarking & CI/CD]
```

---

## 2. Detailed Roadmap Phases

### Phase 1: Unified Deep Learning Runtime & Model Decoupling
- [x] **PyTorch-Native MobileNetV2:** Eliminate fragile TensorFlow Keras dependencies on Windows Python 3.12 by utilizing PyTorch's native `torchvision.models.mobilenet_v2` with ImageNet-1k pretrained weights.
- [x] **Abstract Model Interface (`AbstractVisionModel`):** Define standard contracts for prediction, bounding box extraction, top-k ranking, and confidence scoring.
- [x] **YOLOv8 Modernization:** Wrap Ultralytics YOLOv8 into an enterprise detector class exposing structured bounding box coordinates, IOU thresholds, and category statistics.
- [x] **Dual-Engine Fusion:** Implement a hybrid engine that correlates scene classification with detected objects to identify inconsistencies (e.g., scene says "coffee shop", detected objects are "cup", "laptop", "keyboard").

### Phase 2: Multi-Source Stream Ingestion & Resilient Fallbacks
- [x] **Hardware-Agnostic Stream Adapter:** Support physical webcams (0, 1, ...), video files (`.mp4`, `.avi`), static images (`.jpg`, `.png`), and network streams (RTSP/HTTP).
- [x] **Synthetic Test Pattern Generator:** Generate procedural synthetic frames with moving shapes and colors for automated testing and headless environments without cameras.
- [x] **Auto-Recovery Stream Loop:** Implement reconnection logic and error handlers to prevent silent crashes when a video frame drops.

### Phase 3: Cognitive HUD Overlay & Structured Telemetry
- [x] **Glassmorphic HUD Rendering:** Design high-contrast semi-transparent telemetry boxes displaying current mode, frame resolution, live FPS, and latency breakdown.
- [x] **Category-Aware Bounding Boxes:** Dynamic color mapping based on COCO object categories (vehicles, people, electronics, animals).
- [x] **Precision Latency Metrics:** Measure microsecond-level preprocessing, inference, and rendering times.
- [x] **Telemetry & Media Export:** Enable saving annotated output images, video streams (`cv2.VideoWriter`), and structured JSON detection logs.

### Phase 4: Production Interfaces (REST API & Web Studio)
- [x] **FastAPI REST Microservice:**
  - `POST /api/v1/detect`: YOLOv8 object localization with bounding box coordinates.
  - `POST /api/v1/classify`: MobileNetV2 top-k semantic classification.
  - `POST /api/v1/analyze`: Full dual-engine fusion analysis.
  - `GET /api/v1/health`: System health and hardware/GPU accelerator detection.
- [x] **Streamlit Cognitive Web Studio:**
  - Live model mode switching (MobileNet, YOLO, Fusion).
  - Interactive parameter controls (confidence threshold, IOU threshold, Top-K).
  - Telemetry graphs (latency distribution, FPS gauge, object distribution chart).
  - Side-by-side model comparison viewer.

### Phase 5: Verification, Benchmarking & Packaging
- [x] **Comprehensive Test Suite:** Unit and integration tests covering configuration, model engines, streaming adapters, HUD rendering, and API endpoints.
- [x] **Empirical Benchmark Suite:** Measure throughput (FPS), P50/P90/P99 latency, and memory footprint on synthetic and real sample datasets.
- [x] **Package Distribution:** Modern `pyproject.toml`, `setup.py`, and executable CLI entrypoint `visionpulse`.
