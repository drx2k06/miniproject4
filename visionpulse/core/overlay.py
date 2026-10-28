"""Aesthetic Glassmorphic HUD Graphics & Bounding Box Annotator for VisionPulse."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from visionpulse.models.base import (
    BoundingBox,
    InferenceResult,
    TelemetryData,
)


class HUDOverlay:
    """High-contrast, professional HUD renderer for real-time vision streams."""

    # Curated palette for high visibility across backgrounds
    PALETTE: Dict[str, Tuple[int, int, int]] = {
        "person": (255, 180, 0),       # Cyan-ish Blue in BGR
        "cell phone": (0, 230, 115),   # Neon Green in BGR
        "laptop": (220, 120, 30),      # Vivid Blue in BGR
        "keyboard": (180, 80, 240),    # Magenta in BGR
        "mouse": (200, 200, 0),        # Teal in BGR
        "cup": (40, 150, 255),         # Vivid Amber / Orange in BGR
        "bottle": (100, 220, 255),     # Light Gold in BGR
        "chair": (140, 140, 240),      # Violet in BGR
        "default": (0, 220, 255),      # Bright Yellow-Orange in BGR
    }

    # Mode accent colors (BGR)
    MODE_COLORS: Dict[str, Tuple[int, int, int]] = {
        "detection": (240, 160, 20),      # Ocean Blue / Cyan
        "classification": (50, 190, 255),  # Bright Gold / Amber
        "fusion": (210, 80, 220),          # Neon Purple / Magenta
    }

    @classmethod
    def get_class_color(cls, class_name: str) -> Tuple[int, int, int]:
        """Fetch unique BGR color for class name or derive deterministically."""
        norm = class_name.lower().strip()
        if norm in cls.PALETTE:
            return cls.PALETTE[norm]
        # Hash-based fallback color
        h = abs(hash(norm))
        b = 80 + (h % 150)
        g = 80 + ((h >> 4) % 150)
        r = 80 + ((h >> 8) % 150)
        return (int(b), int(g), int(r))

    @staticmethod
    def _draw_translucent_card(
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        alpha: float = 0.65,
        bg_color: Tuple[int, int, int] = (15, 18, 24),
    ) -> None:
        """Render a rounded translucent background card on the frame."""
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return

        sub_img = image[y1:y2, x1:x2]
        colored_rect = np.full(sub_img.shape, bg_color, dtype=np.uint8)
        cv2.addWeighted(colored_rect, alpha, sub_img, 1.0 - alpha, 0, sub_img)

    @classmethod
    def render_bounding_boxes(
        cls,
        frame: np.ndarray,
        boxes: List[BoundingBox],
    ) -> None:
        """Draw styled bounding boxes with labels and confidence meters."""
        for box in boxes:
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            color = cls.get_class_color(box.class_name)

            # Draw outer box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw corner notch accents for modern aesthetic
            notch = max(10, min(24, int(min(box.width, box.height) * 0.2)))
            thick = 3
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + notch, y1), color, thick)
            cv2.line(frame, (x1, y1), (x1, y1 + notch), color, thick)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - notch, y1), color, thick)
            cv2.line(frame, (x2, y1), (x2, y1 + notch), color, thick)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + notch, y2), color, thick)
            cv2.line(frame, (x1, y2), (x1, y2 - notch), color, thick)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - notch, y2), color, thick)
            cv2.line(frame, (x2, y2), (x2, y2 - notch), color, thick)

            # Draw Label Pill
            label = f"{box.class_name} {box.confidence:.0%}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)

            pill_y1 = max(0, y1 - lh - 10)
            pill_y2 = y1
            pill_x1 = x1
            pill_x2 = x1 + lw + 12

            cls._draw_translucent_card(frame, pill_x1, pill_y1, pill_x2, pill_y2, alpha=0.85, bg_color=color)
            cv2.putText(
                frame,
                label,
                (pill_x1 + 6, pill_y2 - 5),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 0, 0),  # High contrast black text on bright pill
                1,
                cv2.LINE_AA,
            )

    @classmethod
    def render_classification_panel(
        cls,
        frame: np.ndarray,
        result: InferenceResult,
    ) -> None:
        """Render a sleek Top-K classification card with progress bars."""
        if not result.classifications:
            return

        h, w = frame.shape[:2]
        panel_w = min(380, w - 40)
        k = len(result.classifications)
        panel_h = 35 + k * 30

        px1 = 20
        py1 = h - panel_h - 20
        px2 = px1 + panel_w
        py2 = h - 20

        cls._draw_translucent_card(frame, px1, py1, px2, py2, alpha=0.75, bg_color=(20, 24, 32))
        cv2.rectangle(frame, (px1, py1), (px2, py2), (100, 110, 130), 1, cv2.LINE_AA)

        # Panel Title
        cv2.putText(
            frame,
            "TOP PREDICTIONS (ImageNet-1k)",
            (px1 + 12, py1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 210, 220),
            1,
            cv2.LINE_AA,
        )

        for i, pred in enumerate(result.classifications):
            row_y = py1 + 50 + i * 28

            # Label & score
            text = f"{i+1}. {pred.class_name[:20]}"
            score_text = f"{pred.confidence:.1%}"

            cv2.putText(
                frame,
                text,
                (px1 + 12, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                score_text,
                (px2 - 60, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (50, 220, 255),
                1,
                cv2.LINE_AA,
            )

            # Mini Progress Bar
            bar_x1 = px1 + 12
            bar_y1 = row_y + 4
            bar_w = panel_w - 24
            bar_fill = int(bar_w * max(0.0, min(1.0, pred.confidence)))

            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + 4), (50, 55, 65), -1)
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_fill, bar_y1 + 4), (50, 190, 255), -1)

    @classmethod
    def render_hud(
        cls,
        frame: np.ndarray,
        result: InferenceResult,
        telemetry: TelemetryData,
    ) -> np.ndarray:
        """Compose all HUD graphics and telemetry overlays onto the frame."""
        output = frame.copy()
        h, w = output.shape[:2]

        # 1. Top Glassmorphic Navigation Bar
        bar_h = 44
        cls._draw_translucent_card(output, 0, 0, w, bar_h, alpha=0.80, bg_color=(12, 16, 22))
        cv2.line(output, (0, bar_h), (w, bar_h), (60, 70, 85), 1, cv2.LINE_AA)

        # Brand Title
        cv2.putText(
            output,
            "VISIONPULSE",
            (16, 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Mode Indicator Pill
        mode_color = cls.MODE_COLORS.get(result.mode, (180, 180, 180))
        mode_str = f"MODE: {result.mode.upper()}"
        (mw, mh), _ = cv2.getTextSize(mode_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        mx1 = 150
        mx2 = mx1 + mw + 16
        cls._draw_translucent_card(output, mx1, 10, mx2, 34, alpha=0.9, bg_color=mode_color)
        cv2.putText(
            output,
            mode_str,
            (mx1 + 8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (10, 10, 15),
            1,
            cv2.LINE_AA,
        )

        # FPS & Latency Indicator
        fps_color = (0, 230, 115) if telemetry.fps >= 24 else ((0, 200, 255) if telemetry.fps >= 15 else (50, 50, 255))
        fps_str = f"FPS: {telemetry.fps:.1f}"
        cv2.putText(
            output,
            fps_str,
            (mx2 + 25, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            fps_color,
            2,
            cv2.LINE_AA,
        )

        lat_str = f"Latency: {result.latency_ms:.1f}ms ({result.device})"
        cv2.putText(
            output,
            lat_str,
            (mx2 + 120, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 190, 200),
            1,
            cv2.LINE_AA,
        )

        # Resolution indicator on far right
        res_str = f"{w}x{h}"
        cv2.putText(
            output,
            res_str,
            (w - 90, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 160, 170),
            1,
            cv2.LINE_AA,
        )

        # 2. Render Model Annotations
        if result.mode in ("detection", "fusion") and result.boxes:
            cls.render_bounding_boxes(output, result.boxes)

        if result.mode in ("classification", "fusion") and result.classifications:
            cls.render_classification_panel(output, result)

        # 3. Bottom Context Footer (if summary exists)
        if result.context_summary:
            footer_h = 28
            cls._draw_translucent_card(output, 0, h - footer_h, w, h, alpha=0.7, bg_color=(10, 14, 20))
            cv2.putText(
                output,
                result.context_summary[:120],
                (16, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 230, 240),
                1,
                cv2.LINE_AA,
            )

        return output
