"""Empirical benchmark evaluation harness for VisionPulse perception engines."""

import gc
import time
from typing import Any, Dict, List
import numpy as np
import psutil

from visionpulse.benchmarks.test_dataset import SampleDataset
from visionpulse.models.ensemble import DualEngineFusion
from visionpulse.models.mobilenet import MobileNetClassifier
from visionpulse.models.yolo import YOLOv8Detector


class BenchmarkEvaluator:
    """Measures latency distributions, throughput (FPS), and memory footprint."""

    def __init__(self, iterations: int = 15, warmup: int = 3) -> None:
        self.iterations = iterations
        self.warmup = warmup
        self.dataset = SampleDataset()

    def _get_process_ram_mb(self) -> float:
        """Fetch current process resident memory in Megabytes."""
        process = psutil.Process()
        return process.memory_info().rss / (1024.0 * 1024.0)

    def evaluate_model(
        self,
        name: str,
        predict_fn,
        samples: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Run benchmark trials on a model predict function."""
        # Warmup phase
        for _ in range(self.warmup):
            for sample in samples:
                predict_fn(sample)

        latencies_ms: List[float] = []
        ram_before = self._get_process_ram_mb()

        # Measured trials
        for _ in range(self.iterations):
            for sample in samples:
                t0 = time.perf_counter()
                predict_fn(sample)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        ram_after = self._get_process_ram_mb()

        latencies_arr = np.array(latencies_ms)
        mean_lat = float(np.mean(latencies_arr))
        p50_lat = float(np.percentile(latencies_arr, 50))
        p90_lat = float(np.percentile(latencies_arr, 90))
        p99_lat = float(np.percentile(latencies_arr, 99))
        throughput_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        return {
            "model_name": name,
            "trials": len(latencies_ms),
            "mean_latency_ms": round(mean_lat, 2),
            "p50_latency_ms": round(p50_lat, 2),
            "p90_latency_ms": round(p90_lat, 2),
            "p99_latency_ms": round(p99_lat, 2),
            "throughput_fps": round(throughput_fps, 1),
            "ram_delta_mb": round(max(0.0, ram_after - ram_before), 2),
        }

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Execute comprehensive evaluation across MobileNet, YOLOv8, and Dual-Engine."""
        samples_meta = self.dataset.get_test_samples()
        raw_images = [img for _, img, _ in samples_meta]

        if not raw_images:
            raise RuntimeError("No benchmark sample images available.")

        # Initialize engines
        mobilenet = MobileNetClassifier()
        mobilenet.load_model()

        yolo = YOLOv8Detector()
        yolo.load_model()

        fusion = DualEngineFusion(classifier=mobilenet, detector=yolo)
        fusion.load_model()

        results = {}

        # 1. Benchmark MobileNetV2
        results["mobilenet"] = self.evaluate_model(
            "MobileNetV2 (ImageNet-1k)",
            lambda img: mobilenet.predict(img),
            raw_images,
        )

        # 2. Benchmark YOLOv8n
        results["yolo"] = self.evaluate_model(
            "YOLOv8n (COCO)",
            lambda img: yolo.predict(img),
            raw_images,
        )

        # 3. Benchmark DualEngine Fusion
        results["fusion"] = self.evaluate_model(
            "Dual-Engine Fusion",
            lambda img: fusion.predict(img),
            raw_images,
        )

        return results
