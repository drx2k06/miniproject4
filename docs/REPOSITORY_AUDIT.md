# Repository Forensic Audit: Mini-Project 4

**Target Repository:** `https://github.com/kirankumarnscr7-cmyk/mini-project4`  
**Baseline Commit:** `49a9c440639310342d7de0efaeacd470c04cdcc5`  
**Reference Directory:** `$HOME\Projects\miniproject4-reference`  
**Auditor:** Daniel Richard Xavier  
**Date:** October 28, 2026  

---

## 1. Executive Summary

A thorough source-code inspection and forensic analysis of the original `mini-project4` repository was performed. The repository represents a prototype real-time computer vision script (`cnn_imagerec.py`) attempting dual-mode visual inference:
1. **MobileNetV2** ImageNet classification using TensorFlow Keras.
2. **YOLOv8n** multi-class object detection using Ultralytics.

While the conceptual vision (switching between classification and detection) is compelling, the implementation is a raw 49-line script without modularization, input flexibility, testing, documentation, packaging, or error handling.

---

## 2. Baseline Inventory & File Manifest

| File Path | Size | Purpose / Contents | Baseline State |
| :--- | :--- | :--- | :--- |
| `cnn_imagerec.py` | 1,310 bytes (49 lines) | Top-level script executing OpenCV webcam loop with MobileNetV2 and YOLOv8n. | Monolithic, procedural, tightly coupled. |
| `Screenshot 2026-10-15 215747.png` | 419,182 bytes | Screenshot demonstration of MobileNet classification. | Static image asset. |
| `Screenshot 2026-10-15 220102.png` | 535,451 bytes | Screenshot demonstration of YOLOv8 object detection. | Static image asset. |
| `Screenshot 2026-10-15 220419.png` | 537,323 bytes | Screenshot demonstration of detection bounding boxes. | Static image asset. |
| `README.md` | Missing | None | Absent in source. |
| `requirements.txt` | Missing | None | Absent in source. |
| `LICENSE` | Missing | None | Absent in source. |
| `tests/` | Missing | None | Absent in source. |

---

## 3. Deep Code Inspection & Flaws in `cnn_imagerec.py`

### Line-by-Line Breakdown:

- **Lines 1–8: Dual Heavy Framework Imports**
  ```python
  import cv2
  import numpy as np
  from ultralytics import YOLO
  from tensorflow.keras.applications.mobilenet_v2 import (
      MobileNetV2,
      decode_predictions,
      preprocess_input,
  )
  ```
  *Defect:* Unconditionally imports both TensorFlow (Keras) and PyTorch (via Ultralytics) at module startup. On modern Python (e.g. 3.12+), TensorFlow 2.x is frequently incompatible or unavailable on Windows wheels, causing instant crash on launch (`ModuleNotFoundError: No module named 'tensorflow'`). Furthermore, loading both runtimes wastes over 2GB of host RAM.

- **Lines 11–14: Global Model Instantiation & Hardcoded Camera**
  ```python
  classifier = MobileNetV2(weights="imagenet")
  detector = YOLO("yolov8n.pt")
  camera = cv2.VideoCapture(0)
  mode = "mobilenet"
  ```
  *Defect:* Heavy models are instantiated at module scope before any parameter validation. `cv2.VideoCapture(0)` assumes a webcam exists at index `0`. In headless environments, Docker containers, cloud VMs, or workstations without a connected camera, `camera.isOpened()` returns `False` or raises an exception, causing the program to immediately terminate silently.

- **Lines 16–20: Brittle Frame Ingestion Loop**
  ```python
  while camera.isOpened():
      ok, frame = camera.read()
      if not ok:
          break
  ```
  *Defect:* If a single frame read fails (common during camera re-initialization or stream hiccup), the entire program exits silently without logging any diagnostic information.

- **Lines 21–22: YOLO Mode Execution**
  ```python
  if mode == "yolo":
      frame = detector(frame, verbose=False)[0].plot()
  ```
  *Defect:* Directly invokes `.plot()` which draws default Ultralytics styling without exposing raw bounding boxes, confidence scores, class labels, or count statistics. The application cannot extract structured data or telemetry.

- **Lines 23–37: MobileNet Classification Execution**
  ```python
  else:
      image = cv2.resize(frame, (224, 224))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = preprocess_input(np.expand_dims(image.astype(np.float32), axis=0))
      predictions = classifier.predict(image, verbose=0)
      _, name, score = decode_predictions(predictions, top=1)[0][0]
      cv2.putText(
          frame,
          f"{name}: {score:.0%}",
          (10, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          1,
          (0, 255, 0),
          2,
      )
  ```
  *Defect:* Performs synchronous inference per frame with no latency measurement or frame skipping. Only displays top-1 prediction; if the top-1 confidence is 12% (a poor classification), it still displays with no threshold filtering. Raw `cv2.putText` draws unpadded text that frequently overlaps image edges or is unreadable against bright backgrounds.

- **Lines 39–49: Display and Cleanup**
  ```python
  cv2.imshow("m=MobileNet, y=YOLO, q=quit", frame)
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
      break
  ...
  camera.release()
  cv2.destroyAllWindows()
  ```
  *Defect:* `cv2.imshow` crashes on headless servers or virtual displays (`error: The function is not implemented`). No headless mode, file output mode, or streaming web mode exists.

---

## 4. Key Engineering Debt & Risk Matrix

| Risk / Defect | Severity | Impact | Mitigation in VisionPulse |
| :--- | :--- | :--- | :--- |
| **TensorFlow Python 3.12 Incompatibility** | **HIGH** | Script fails to import on modern Python environments. | Re-engineered with native PyTorch MobileNetV2 (with optional TensorFlow fallback), ensuring unified PyTorch runtime across both models. |
| **Hardcoded VideoCapture(0)** | **HIGH** | Script crashes or terminates on machines without webcams. | Multi-source streaming adapter supporting webcams, video files, static images, RTSP streams, and synthetic procedural test pattern generation. |
| **Monolithic Architecture** | **HIGH** | Zero testability, reusability, or maintainability. | Modular OOP design (`visionpulse.models`, `visionpulse.core`, `visionpulse.benchmarks`, `visionpulse.web`). |
| **Zero Input Validation / Error Handling** | **HIGH** | Unhandled exceptions and silent crashes. | Strict Pydantic configuration, safe frame acquisition with fallback handlers, and structured exception classes. |
| **No Benchmark or Evaluation Metric** | **MEDIUM** | Unknown latency, throughput, or accuracy. | Built-in evaluation harness measuring P50/P90/P99 latency, FPS throughput, memory footprint, and detection accuracy. |
| **Absence of User Interface** | **MEDIUM** | Restricted to local desktop OpenCV window. | Interactive Streamlit Web Studio and production-grade FastAPI REST endpoints. |
| **Lack of Packaging & Dependencies** | **MEDIUM** | Inability to install or distribute via pip. | Standard `pyproject.toml`, `setup.py`, `requirements.txt`, and entrypoint CLI `visionpulse`. |

---

## 5. Conclusion & Modernization Mandate

The original repository represents a valuable dual-paradigm concept (Classification vs. Detection) that requires full enterprise re-engineering. The modernized project—**VisionPulse**—retains the dual-engine vision while transforming it into a high-performance, modular, testable, and cross-platform vision intelligence platform.
