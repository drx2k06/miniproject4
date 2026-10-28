"""Structured exporter for telemetry logs, annotated images, and video recordings."""

import json
import os
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from visionpulse.models.base import InferenceResult, TelemetryData


class DataExporter:
    """Manages storage of vision telemetry, image snapshots, and video streams."""

    def __init__(self, export_dir: str = "./output") -> None:
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._recording_path: Optional[str] = None

    def save_image(
        self,
        frame: np.ndarray,
        filename: Optional[str] = None,
    ) -> str:
        """Save a single image frame to the export directory.

        Returns:
            Absolute or relative file path of the saved image.
        """
        if filename is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{ts}.jpg"

        filepath = os.path.join(self.export_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def save_telemetry_log(
        self,
        result: InferenceResult,
        telemetry: TelemetryData,
        filename: Optional[str] = None,
    ) -> str:
        """Serialize inference and telemetry results to formatted JSON."""
        if filename is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"telemetry_{ts}_{telemetry.frame_idx}.json"

        filepath = os.path.join(self.export_dir, filename)

        data = {
            "timestamp": result.timestamp.isoformat(),
            "model_name": result.model_name,
            "mode": result.mode,
            "device": result.device,
            "latency_ms": result.latency_ms,
            "telemetry": telemetry.model_dump(),
            "classifications": [c.model_dump() for c in result.classifications],
            "detections": [b.model_dump() for b in result.boxes],
            "counts": result.counts,
            "context_summary": result.context_summary,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return filepath

    def start_video_recording(
        self,
        filename: str = "recording.mp4",
        fps: int = 30,
        resolution: Tuple[int, int] = (640, 480),
        fourcc: str = "mp4v",
    ) -> str:
        """Initialize video recording writer."""
        if self._video_writer is not None:
            self.stop_video_recording()

        self._recording_path = os.path.join(self.export_dir, filename)
        codec = cv2.VideoWriter_fourcc(*fourcc)
        self._video_writer = cv2.VideoWriter(self._recording_path, codec, float(fps), resolution)
        return self._recording_path

    def write_video_frame(self, frame: np.ndarray) -> None:
        """Write frame to active video recording."""
        if self._video_writer is not None and self._video_writer.isOpened():
            self._video_writer.write(frame)

    def stop_video_recording(self) -> Optional[str]:
        """Finalize and close active video writer."""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            path = self._recording_path
            self._recording_path = None
            return path
        return None
