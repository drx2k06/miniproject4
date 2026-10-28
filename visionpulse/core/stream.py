"""Multi-source stream ingestion adapter supporting cameras, video, images, and synthetic patterns."""

import math
import os
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np


class StreamSource:
    """Robust video and image ingestion adapter."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        loop: bool = True,
        synthetic_fps: int = 30,
        synthetic_res: Tuple[int, int] = (640, 480),
    ) -> None:
        """Initialize StreamSource.

        Args:
            source: Camera index (int), file path (str), or 'synthetic'/'test'.
            loop: Whether to loop video files when reaching end of file.
            synthetic_fps: Target FPS for synthetic generator.
            synthetic_res: (width, height) resolution for synthetic generator.
        """
        self.source = source
        self.loop = loop
        self.synthetic_fps = synthetic_fps
        self.synthetic_res = synthetic_res

        self.source_type: str = "unknown"
        self._cap: Optional[cv2.VideoCapture] = None
        self._static_image: Optional[np.ndarray] = None
        self._frame_count = 0
        self._start_time = time.time()
        self._is_opened = False

        self._init_source()

    def _init_source(self) -> None:
        """Inspect and initialize underlying capture device or generator."""
        if str(self.source).lower() in ("synthetic", "test", "dummy"):
            self.source_type = "synthetic"
            self._is_opened = True
            return

        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            cam_idx = int(self.source)
            self.source_type = "webcam"
            self._cap = cv2.VideoCapture(cam_idx)
            self._is_opened = self._cap.isOpened()
            return

        path = str(self.source)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input source path does not exist: {path}")

        # Check if static image
        lower = path.lower()
        if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            self.source_type = "image"
            self._static_image = cv2.imread(path)
            if self._static_image is None:
                raise ValueError(f"Failed to decode image file: {path}")
            self._is_opened = True
            return

        # Treat as video file
        self.source_type = "video"
        self._cap = cv2.VideoCapture(path)
        self._is_opened = self._cap.isOpened()

    @property
    def is_opened(self) -> bool:
        """Return whether stream is successfully opened."""
        return self._is_opened

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return (width, height) resolution of the stream."""
        if self.source_type == "synthetic":
            return self.synthetic_res
        if self.source_type == "image" and self._static_image is not None:
            return (self._static_image.shape[1], self._static_image.shape[0])
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (0, 0)

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Procedurally render a synthetic animated frame for testing and headless CI."""
        w, h = self.synthetic_res
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Dynamic gradient background
        t = self._frame_count * 0.05
        for y in range(0, h, 10):
            val = int(127 + 127 * math.sin(t + y * 0.02))
            frame[y : y + 10, :, 0] = val // 3
            frame[y : y + 10, :, 1] = val // 2
            frame[y : y + 10, :, 2] = val // 4

        # Moving geometric target object (simulates a moving object to detect)
        obj_x = int((w / 2) + (w / 3) * math.sin(t))
        obj_y = int((h / 2) + (h / 4) * math.cos(t * 1.5))
        obj_size = 60

        # Draw a simulated entity
        cv2.rectangle(
            frame,
            (obj_x - obj_size, obj_y - obj_size),
            (obj_x + obj_size, obj_y + obj_size),
            (0, 220, 255),
            -1,
        )
        cv2.circle(frame, (obj_x, obj_y), obj_size // 2, (50, 50, 255), -1)

        # Overlay test pattern metadata
        text = f"VisionPulse Synthetic Source | Frame #{self._frame_count}"
        cv2.putText(
            frame,
            text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Fetch next frame from underlying source."""
        if not self._is_opened:
            return False, None

        self._frame_count += 1

        if self.source_type == "synthetic":
            # Simulate framerate pacing
            time.sleep(1.0 / max(1, self.synthetic_fps))
            return True, self._generate_synthetic_frame()

        if self.source_type == "image":
            # Static image yields a fresh copy of the image
            return True, self._static_image.copy()

        if self._cap and self._cap.isOpened():
            ok, frame = self._cap.read()
            if not ok:
                if self.loop and self.source_type == "video":
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._cap.read()
                    return ok, frame
                return False, None
            return True, frame

        return False, None

    def release(self) -> None:
        """Release capture device and free allocated resources."""
        if self._cap:
            self._cap.release()
            self._cap = None
        self._is_opened = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
