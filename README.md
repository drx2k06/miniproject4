# VisionPulse: Dual-Engine Real-Time Computer Vision & Cognitive Object Intelligence System

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Studio-ff4b4b.svg)](https://streamlit.io)
[![Test Suite](https://img.shields.io/badge/Tests-27%20Passed%20(100%25)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Lead Architect & Developer:** [Daniel Richard Xavier](https://github.com/drx2k06)  
> **Original Concept Baseline:** [kirankumarnscr7-cmyk/mini-project4](https://github.com/kirankumarnscr7-cmyk/mini-project4)  

---

## 1. Overview & Problem Statement

Conventional real-time computer vision scripts often force an artificial choice between whole-scene semantic classification (what kind of environment is this?) and local spatial bounding-box detection (where are the individual items in this environment?).

The original proof-of-concept repository (`mini-project4`) attempted to bridge this divide through a 49-line OpenCV script (`cnn_imagerec.py`) that alternated between a TensorFlow Keras MobileNetV2 classifier and an Ultralytics YOLOv8 detector. However, the original code had critical engineering bottlenecks:
1. **Broken Runtimes on Modern Python:** Depended on TensorFlow 2.x, which fails to install or compile on many Windows Python 3.12 installations.
2. **Hardcoded VideoCapture(0):** Crashed immediately on machines without a physical webcam at index 0 (e.g. servers, virtual machines, cloud instances).
3. **Monolithic Architecture:** Contained zero modularity, zero test coverage, zero input validation, and zero headless CI/CD capability.
4. **Lack of Integration & Telemetry:** No latency tracking, no FPS gauges, no output persistence, and no REST or web interfaces.

**VisionPulse** completely re-engineers this foundation into an enterprise-grade cognitive perception platform, unifying PyTorch-native MobileNetV2 and Ultralytics YOLOv8 within a thread-safe, multi-source ingestion pipeline backed by an aesthetic glassmorphic HUD graphics engine, a FastAPI REST microservice, and a Streamlit Web Studio.

---

## 2. Key Features

- **⚡ Unified PyTorch Runtime:**
  MobileNetV2 has been rewritten from TensorFlow to native PyTorch (`torchvision.models.mobilenet_v2`) with ImageNet-1k weights. This cuts memory usage by over 1.2 GB and guarantees 100% compatibility across Python 3.10–3.12+ on Windows, Linux, and macOS.
- **🎯 Dual-Engine Cognitive Fusion:**
  Combines scene-level classification with multi-object bounding-box localization to generate holistic cognitive summaries (e.g., *"Scene context: 'Office' (92.1%). Localized entities: [1 laptop, 1 cup, 1 keyboard]"*).
- **📹 Multi-Source Stream Ingestion:**
  Ingests live webcams (`0`, `1`), video files (`.mp4`, `.avi`), static images (`.jpg`, `.png`), RTSP streams, and an animated **procedural synthetic test pattern generator** for headless automated testing without camera hardware.
- **🎨 Glassmorphic HUD & Annotations:**
  Real-time graphic overlays featuring translucent telemetry cards, live FPS meters, latency breakdowns (pre/inference/post), category-coded bounding box corner notches, and horizontal Top-K probability bars.
- **🌐 Dual Modern Interfaces:**
  - **FastAPI REST Microservice:** Async endpoints for object detection, classification, ensemble analysis, and system diagnostics with automatic OpenAPI Swagger documentation.
  - **Streamlit Cognitive Studio:** Interactive dashboard featuring parameter sliders, Plotly telemetry charts, side-by-side model comparison, and JSON export.
- **💾 Structured Telemetry & Recording:**
  Exports microsecond-precision JSON telemetry logs, high-resolution annotated snapshots, and MP4 video recordings.
- **🧪 100% Verified Automated Test Suite:**
  27 comprehensive unit and integration tests covering models, streaming adapters, HUD rendering, pipelines, and REST APIs.

---

## 3. Architecture & System Flow

```mermaid
flowchart TD
    subgraph Inputs["Multi-Source Ingestion"]
        Webcam["Physical Webcam (0, 1)"]
        Video["Video File (.mp4, .avi)"]
        Image["Static Image (.jpg, .png)"]
        Synthetic["Synthetic Generator (CI / Headless)"]
    end

    subgraph Core["VisionPulse Core Pipeline"]
        Adapter["StreamSource Adapter"]
        Engine["VisionPipeline"]
        Overlay["HUDOverlay (Glassmorphic Graphics)"]
        Exporter["DataExporter (JSON / Image / Video)"]
    end

    subgraph Models["Unified Perception Layer"]
        MNet["MobileNetV2 (PyTorch ImageNet-1k)"]
        Yolo["YOLOv8n (Ultralytics COCO)"]
        Fusion["Dual-Engine Fusion Ensemble"]
    end

    subgraph Outputs["Presentation Layer"]
        CLI["Rich Terminal CLI"]
        API["FastAPI REST Microservice"]
        Studio["Streamlit Web Studio"]
    end

    Inputs --> Adapter
    Adapter --> Engine
    Engine --> Models
    Models --> Overlay
    Overlay --> Exporter
    Exporter --> Outputs
```

For detailed architectural specifications and sequence diagrams, refer to [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 4. Repository Structure

```
miniproject4/
├── .env.example              # Safe environment variable configuration template
├── .gitignore                # Enterprise-grade Git ignore manifest
├── LICENSE                   # MIT License with author attribution
├── README.md                 # Primary system documentation
├── pyproject.toml            # Modern PEP 518/621 build specification
├── requirements.txt          # Production dependency manifest
├── requirements-dev.txt      # Development & testing dependencies
├── setup.py                  # Package installation entrypoint
├── data/
│   └── samples/              # High-contrast benchmark sample imagery
├── docs/
│   ├── ARCHITECTURE.md       # Full architecture & sequence diagrams
│   ├── CHANGELOG.md          # Version history and migration notes
│   ├── EVALUATION_REPORT.md  # Empirical latency & throughput benchmarks
│   ├── FINAL_REPORT.md       # Comprehensive engineering verification report
│   ├── IMPROVEMENT_ROADMAP.md# Strategic 5-phase engineering roadmap
│   └── REPOSITORY_AUDIT.md   # Line-by-line forensics of original source
├── legacy/
│   ├── cnn_imagerec.py       # Original author's baseline implementation
│   └── Screenshot *.png      # Original baseline demonstration assets
├── tests/
│   ├── test_api.py           # FastAPI REST endpoints tests
│   ├── test_benchmarks.py    # Benchmark harness and dataset tests
│   ├── test_config.py        # Configuration and validation tests
│   ├── test_models.py        # MobileNet, YOLOv8, and Fusion tests
│   ├── test_overlay.py       # HUD graphics and styling tests
│   ├── test_pipeline.py      # Core pipeline execution tests
│   └── test_stream.py        # Ingestion and synthetic generator tests
└── visionpulse/
    ├── __init__.py           # Package exports and version metadata
    ├── cli.py                # Command-line interface entrypoint
    ├── config.py             # Pydantic Settings configuration manager
    ├── benchmarks/           # Empirical benchmark evaluation harness
    ├── core/                 # Ingestion, pipeline, HUD, and exporter
    ├── models/               # PyTorch MobileNet, YOLOv8, and Fusion
    └── web/                  # FastAPI microservice and Streamlit Studio
```

---

## 5. Prerequisites & Environment Setup

### Prerequisites
- Windows 10/11, macOS, or Linux
- Python 3.10, 3.11, or 3.12 (Python 3.12.10 verified)
- Git 2.40+
- (Optional) NVIDIA GPU with CUDA 11.8+ / 12.1+ for accelerated inference. Automatically defaults to multi-threaded CPU if no GPU is found.

### Verified Windows PowerShell Installation

```powershell
# 1. Clone repository
git clone https://github.com/drx2k06/miniproject4.git
Set-Location miniproject4

# 2. (Optional) Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Install package in editable development mode
python -m pip install -e .
```

---

## 6. Execution & Quickstart Guide

### 6.1 Interactive Stream (Desktop Window)
Run live video perception with keyboard mode switching:
```bash
# Ingest from webcam 0
visionpulse run --source 0 --mode detection

# Ingest from a video file
visionpulse run --source "path/to/video.mp4" --mode fusion

# Ingest using synthetic test pattern (works without a physical camera)
visionpulse run --source synthetic --mode fusion
```

**Keyboard Shortcuts during live stream:**
- `m`: Switch to **MobileNetV2 Classification**
- `y`: Switch to **YOLOv8 Object Detection**
- `f`: Switch to **Dual-Engine Fusion**
- `s`: Save high-resolution annotated snapshot and JSON telemetry
- `r`: Start / Stop video recording to MP4
- `q`: Quit and release resources

### 6.2 Streamlit Cognitive Web Studio
Launch the interactive web browser dashboard:
```bash
visionpulse studio --port 8501
```
Open `http://localhost:8501` to adjust confidence thresholds, inspect Plotly telemetry charts, and run side-by-side model comparisons.

### 6.3 FastAPI REST Microservice
Start the production REST microservice:
```bash
visionpulse serve --host 127.0.0.1 --port 8000
```
Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

**Example curl command to run detection:**
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/detect?conf_threshold=0.35" \
  -H "accept: application/json" \
  -F "file=@data/samples/office_scene.jpg"
```

### 6.4 Single-Image Command Line Inference
```bash
# Infer with JSON output
visionpulse infer --input data/samples/office_scene.jpg --mode fusion --json

# Infer and save annotated image output
visionpulse infer --input data/samples/traffic_scene.jpg --mode detection --output output/annotated.jpg
```

---

## 7. Empirical Evaluation & Benchmarks

Benchmarking was conducted on an x86_64 Windows host running Python 3.12 under multi-threaded CPU execution across 15 measured trials with 3 warmup cycles:

| Model / Engine | Task Type | Resolution | P50 Latency (ms) | P90 Latency (ms) | Throughput (FPS) | RAM Delta (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | Semantic Classification | 224x224 | **23.64 ms** | **25.28 ms** | **42.2 FPS** | ~0.01 MB |
| **YOLOv8n** | Multi-Class Object Detection | 640x480 | **52.14 ms** | **55.00 ms** | **19.0 FPS** | ~0.00 MB |
| **Dual-Engine Fusion** | Joint Cognitive Perception | Dynamic | **76.01 ms** | **80.88 ms** | **13.1 FPS** | ~0.00 MB |

To replicate these benchmark numbers:
```bash
visionpulse benchmark --iterations 15 --warmup 3
```
For the complete analysis, view [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md).

---

## 8. Verification & Automated Testing

Execute the full automated test suite:
```bash
python -m pytest
```

**Verification Results:**
```
tests/test_api.py ......                                                 [ 22%]
tests/test_benchmarks.py ..                                              [ 29%]
tests/test_config.py ....                                                [ 44%]
tests/test_models.py ......                                              [ 66%]
tests/test_overlay.py ..                                                 [ 74%]
tests/test_pipeline.py ....                                              [ 88%]
tests/test_stream.py ...                                                 [100%]

====================== 27 passed, 0 warnings in 10.97s =======================
```

---

## 9. Security, Privacy & Dependency Audit

- **Zero Hardcoded Secrets:** All configurations and endpoints use safe defaults and environment variables via `.env.example`.
- **Safe Input Decoding:** Image upload buffers in FastAPI and Streamlit validate headers and dimensions before decoding to prevent memory exhaustion (DoS).
- **Graceful Headless Execution:** Display calls are guarded against headless terminal environments, preventing GUI segmentation faults.
- **Licensing Compliance:** Preserves attribution to original author kirankumarnscr7-cmyk under the permissive MIT license.

---

## 10. Authorship & Engineering Credits

- **Re-Engineering & Customization Specialist:**  
  **Daniel Richard Xavier** ([drx2023l@gmail.com](mailto:drx2023l@gmail.com))  
  *Responsibilities: Architecture redesign, PyTorch MobileNetV2 migration, stream abstraction, HUD graphics engine, FastAPI backend, Streamlit Studio, test suite, and evaluation benchmarking.*

- **Original Baseline Author:**  
  **kirankumarnscr7-cmyk** ([mini-project4](https://github.com/kirankumarnscr7-cmyk/mini-project4))  
  *Baseline commit `49a9c440639310342d7de0efaeacd470c04cdcc5` preserved in `legacy/`.*

---

## 11. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full details.
