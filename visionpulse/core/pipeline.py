"""Real-time processing pipeline orchestrator for VisionPulse."""

import os
import time
from typing import Callable, Optional, Tuple
import cv2
import numpy as np

from visionpulse.config import VisionConfig, get_config
from visionpulse.core.exporter import DataExporter
from visionpulse.core.overlay import HUDOverlay
from visionpulse.core.stream import StreamSource
from visionpulse.models.base import InferenceResult, TelemetryData
from visionpulse.models.ensemble import DualEngineFusion
from visionpulse.models.mobilenet import MobileNetClassifier
from visionpulse.models.yolo import YOLOv8Detector


class VisionPipeline:
    """Core real-time vision execution pipeline."""

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        mode: str = "detection",  # 'detection', 'classification', 'fusion'
    ) -> None:
        self.config = config or get_config()
        self.mode = mode.lower()

        # Instantiate models lazily
        self._classifier: Optional[MobileNetClassifier] = None
        self._detector: Optional[YOLOv8Detector] = None
        self._fusion: Optional[DualEngineFusion] = None

        self.exporter = DataExporter(export_dir=self.config.export.export_dir)
        self.overlay = HUDOverlay()

        # Telemetry metrics
        self.frame_idx = 0
        self._fps = 0.0
        self._prev_time = time.perf_counter()
        self._is_recording = False

    @property
    def classifier(self) -> MobileNetClassifier:
        if self._classifier is None:
            self._classifier = MobileNetClassifier(
                device=self.config.models.device,
                top_k=self.config.models.mobilenet_top_k,
            )
        return self._classifier

    @property
    def detector(self) -> YOLOv8Detector:
        if self._detector is None:
            self._detector = YOLOv8Detector(
                model_name_or_path=self.config.models.yolo_model,
                device=self.config.models.device,
                conf_threshold=self.config.models.yolo_conf_threshold,
                iou_threshold=self.config.models.yolo_iou_threshold,
            )
        return self._detector

    @property
    def fusion(self) -> DualEngineFusion:
        if self._fusion is None:
            self._fusion = DualEngineFusion(
                classifier=self.classifier,
                detector=self.detector,
                device=self.config.models.device,
            )
        return self._fusion

    def set_mode(self, mode: str) -> None:
        """Switch active perception mode: 'detection', 'classification', or 'fusion'."""
        valid_modes = ("detection", "classification", "fusion")
        clean_mode = mode.lower().strip()
        if clean_mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}")
        self.mode = clean_mode

    def process_frame(
        self,
        frame: np.ndarray,
        mode: Optional[str] = None,
        conf_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, InferenceResult, TelemetryData]:
        """Process a single frame: inference, telemetry calculation, and HUD rendering."""
        current_mode = (mode or self.mode).lower()
        self.frame_idx += 1

        t0 = time.perf_counter()

        # 1. Inference dispatch
        if current_mode == "classification":
            result = self.classifier.predict(frame)
        elif current_mode == "fusion":
            result = self.fusion.predict(frame, conf_threshold=conf_threshold)
        else:  # detection
            result = self.detector.predict(frame, conf_threshold=conf_threshold)

        t1 = time.perf_counter()
        inference_time_ms = (t1 - t0) * 1000.0

        # 2. FPS Calculation (exponential moving average)
        now = time.perf_counter()
        delta = now - self._prev_time
        self._prev_time = now
        instant_fps = 1.0 / max(1e-5, delta)
        self._fps = instant_fps if self._fps == 0.0 else (0.85 * self._fps + 0.15 * instant_fps)

        h, w = frame.shape[:2]
        telemetry = TelemetryData(
            frame_idx=self.frame_idx,
            fps=self._fps,
            preprocess_ms=0.5,
            inference_ms=inference_time_ms,
            postprocess_ms=0.5,
            total_latency_ms=inference_time_ms + 1.0,
            resolution=(w, h),
            device=result.device,
        )

        # 3. Render HUD
        annotated_frame = self.overlay.render_hud(frame, result, telemetry)

        # 4. Optional video recording
        if self._is_recording:
            self.exporter.write_video_frame(annotated_frame)

        return annotated_frame, result, telemetry

    def run_live(
        self,
        stream_source: StreamSource,
        display: bool = True,
        max_frames: Optional[int] = None,
        on_frame_callback: Optional[Callable[[np.ndarray, InferenceResult, TelemetryData], None]] = None,
    ) -> None:
        """Run interactive streaming loop with keyboard controls."""
        window_name = "VisionPulse Studio | [m]obileNet [y]olo [f]usion [s]napshot [r]ecord [q]uit"

        # Check for headless environment
        is_headless = not display or os.environ.get("HEADLESS", "0") == "1"

        try:
            while stream_source.is_opened:
                if max_frames and self.frame_idx >= max_frames:
                    break

                ok, frame = stream_source.read()
                if not ok or frame is None:
                    break

                annotated_frame, result, telemetry = self.process_frame(frame)

                if on_frame_callback:
                    on_frame_callback(annotated_frame, result, telemetry)

                if not is_headless:
                    cv2.imshow(window_name, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break
                    elif key == ord("m"):
                        self.set_mode("classification")
                    elif key == ord("y"):
                        self.set_mode("detection")
                    elif key == ord("f"):
                        self.set_mode("fusion")
                    elif key == ord("s"):
                        path = self.exporter.save_image(annotated_frame)
                        self.exporter.save_telemetry_log(result, telemetry)
                        print(f"Snapshot & telemetry saved to: {path}")
                    elif key == ord("r"):
                        if not self._is_recording:
                            res = (annotated_frame.shape[1], annotated_frame.shape[0])
                            self.exporter.start_video_recording(resolution=res)
                            self._is_recording = True
                            print("Started video recording...")
                        else:
                            path = self.exporter.stop_video_recording()
                            self._is_recording = False
                            print(f"Stopped video recording. Saved to: {path}")

        finally:
            stream_source.release()
            if self._is_recording:
                self.exporter.stop_video_recording()
            if not is_headless:
                cv2.destroyAllWindows()
