"""Sample dataset generator and ground-truth metadata for benchmark evaluation."""

import os
from typing import Dict, List, Tuple
import cv2
import numpy as np


class SampleDataset:
    """Manages benchmark test images and ground truth annotations."""

    def __init__(self, data_dir: str = "./data/samples") -> None:
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self._ensure_sample_images()

    def _ensure_sample_images(self) -> None:
        """Generate high-contrast, labeled benchmark images if not present."""
        office_path = os.path.join(self.data_dir, "office_scene.jpg")
        if not os.path.exists(office_path):
            img = self._generate_office_scene()
            cv2.imwrite(office_path, img)

        traffic_path = os.path.join(self.data_dir, "traffic_scene.jpg")
        if not os.path.exists(traffic_path):
            img = self._generate_traffic_scene()
            cv2.imwrite(traffic_path, img)

        living_path = os.path.join(self.data_dir, "living_room.jpg")
        if not os.path.exists(living_path):
            img = self._generate_living_room_scene()
            cv2.imwrite(living_path, img)

    def _generate_office_scene(self) -> np.ndarray:
        """Synthesize an office workspace scene (table, laptop, cup, notebook)."""
        w, h = 800, 600
        img = np.full((h, w, 3), (220, 220, 225), dtype=np.uint8)  # Office wall

        # Wooden desk surface
        cv2.rectangle(img, (0, 320), (w, h), (90, 130, 170), -1)

        # Laptop (body, screen, keyboard)
        # Base
        cv2.rectangle(img, (240, 360), (560, 480), (120, 120, 125), -1)
        # Screen
        cv2.rectangle(img, (260, 180), (540, 360), (40, 40, 45), -1)
        cv2.rectangle(img, (270, 190), (530, 350), (230, 180, 70), -1)  # Display screen
        cv2.putText(img, "VisionPulse Code", (290, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Coffee Cup
        cv2.rectangle(img, (140, 380), (200, 460), (255, 255, 255), -1)
        cv2.ellipse(img, (200, 420), (15, 25), 0, -90, 90, (255, 255, 255), 4)  # Handle

        # Cell Phone
        cv2.rectangle(img, (600, 400), (660, 490), (30, 30, 30), -1)
        cv2.rectangle(img, (605, 410), (655, 480), (80, 80, 80), -1)

        return img

    def _generate_traffic_scene(self) -> np.ndarray:
        """Synthesize an outdoor street scene with vehicles."""
        w, h = 800, 600
        img = np.full((h, w, 3), (240, 200, 160), dtype=np.uint8)  # Sky

        # Road
        cv2.rectangle(img, (0, 300), (w, h), (60, 60, 65), -1)
        # Road lane stripes
        for x in range(50, w, 150):
            cv2.rectangle(img, (x, 440), (x + 80, 455), (255, 255, 255), -1)

        # Blue Car body
        cv2.rectangle(img, (250, 350), (550, 440), (220, 80, 30), -1)
        cv2.rectangle(img, (310, 280), (480, 350), (220, 80, 30), -1)
        # Wheels
        cv2.circle(img, (310, 440), 30, (20, 20, 20), -1)
        cv2.circle(img, (490, 440), 30, (20, 20, 20), -1)

        return img

    def _generate_living_room_scene(self) -> np.ndarray:
        """Synthesize an indoor living room scene."""
        w, h = 800, 600
        img = np.full((h, w, 3), (210, 220, 215), dtype=np.uint8)

        # Floor
        cv2.rectangle(img, (0, 380), (w, h), (120, 100, 80), -1)

        # Sofa / Couch
        cv2.rectangle(img, (180, 320), (620, 460), (80, 50, 180), -1)
        cv2.rectangle(img, (150, 340), (200, 440), (70, 40, 160), -1)  # Left arm
        cv2.rectangle(img, (600, 340), (650, 440), (70, 40, 160), -1)  # Right arm

        # Potted Plant / Bottle on table
        cv2.rectangle(img, (80, 360), (120, 440), (40, 140, 60), -1)

        return img

    def get_test_samples(self) -> List[Tuple[str, np.ndarray, Dict[str, any]]]:
        """Return list of (sample_name, image_array, ground_truth_meta)."""
        samples = []
        for name in ["office_scene.jpg", "traffic_scene.jpg", "living_room.jpg"]:
            path = os.path.join(self.data_dir, name)
            if os.path.exists(path):
                img = cv2.imread(path)
                samples.append((name, img, {"path": path}))
        return samples
