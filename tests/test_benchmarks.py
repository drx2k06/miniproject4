"""Unit tests for BenchmarkEvaluator and SampleDataset."""

import numpy as np
from visionpulse.benchmarks.evaluator import BenchmarkEvaluator
from visionpulse.benchmarks.test_dataset import SampleDataset


def test_sample_dataset_generation(tmp_path):
    ds = SampleDataset(data_dir=str(tmp_path / "samples"))
    samples = ds.get_test_samples()
    assert len(samples) >= 3
    for name, img, meta in samples:
        assert isinstance(img, np.ndarray)
        assert img.shape[0] > 0
        assert img.shape[1] > 0


def test_benchmark_evaluator_execution():
    evaluator = BenchmarkEvaluator(iterations=2, warmup=1)
    results = evaluator.run_full_benchmark()

    assert "mobilenet" in results
    assert "yolo" in results
    assert "fusion" in results

    for key, metrics in results.items():
        assert "mean_latency_ms" in metrics
        assert "p50_latency_ms" in metrics
        assert "throughput_fps" in metrics
        assert metrics["p50_latency_ms"] > 0
        assert metrics["throughput_fps"] > 0
