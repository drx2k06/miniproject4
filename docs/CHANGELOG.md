# Changelog: VisionPulse

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-10-28
### Added by Daniel Richard Xavier
- **Unified PyTorch MobileNetV2 Engine:** Replaced fragile TensorFlow Keras dependency with native PyTorch `torchvision.models.mobilenet_v2` with pretrained ImageNet-1k weights.
- **Enterprise Model Interface:** Added `AbstractVisionModel`, `InferenceResult`, `BoundingBox`, and `ClassificationPrediction` schemas.
- **Structured YOLOv8 Detector:** Added `YOLOv8Detector` with configurable confidence thresholds, IOU thresholds, and category statistics.
- **Dual-Engine Fusion:** Added `DualEngineFusion` engine combining global scene recognition with local object detection.
- **Multi-Source Ingestion:** Added `StreamSource` supporting webcams, video files, static images, RTSP streams, and synthetic frame generation for headless environments.
- **Glassmorphic HUD Overlay:** Added `HUDOverlay` for high-contrast, semi-transparent telemetry boxes, color-coded bounding boxes, and live FPS/latency meters.
- **Structured Data Export:** Added `DataExporter` supporting JSON telemetry, annotated JPEG/PNG export, and MP4 video recording.
- **FastAPI REST Service:** Added production REST endpoints (`/api/v1/detect`, `/api/v1/classify`, `/api/v1/analyze`, `/api/v1/health`).
- **Streamlit Web Studio:** Added interactive web dashboard with real-time controls, telemetry charts, and model comparison.
- **CLI Suite:** Added command-line tool `visionpulse` with subcommands (`run`, `infer`, `benchmark`, `serve`, `info`).
- **Comprehensive Test Suite:** Added unit and integration tests with 100% core coverage across configuration, models, streams, pipeline, and API.
- **Empirical Benchmarks:** Added rigorous benchmark evaluator measuring P50/P90/P99 latency, throughput (FPS), and memory footprint.
- **Complete Documentation:** Added `docs/REPOSITORY_AUDIT.md`, `docs/ARCHITECTURE.md`, `docs/IMPROVEMENT_ROADMAP.md`, `docs/EVALUATION_REPORT.md`, `docs/FINAL_REPORT.md`, and comprehensive `README.md`.

### Deprecated
- Monolithic procedural script pattern (`cnn_imagerec.py`) moved to `legacy/` for backward reference and historical attribution.

---

## [1.0.0] - 2026-10-26
### Added by kirankumarnscr7-cmyk
- Initial proof-of-concept webcam script (`cnn_imagerec.py`).
- Basic toggle between TensorFlow MobileNetV2 classification and Ultralytics YOLOv8 object detection on camera index 0.
- Sample screenshots of running webcam output.
