"""Command Line Interface (CLI) for VisionPulse."""

import argparse
import json
import os
import sys
import cv2
import uvicorn

from visionpulse import __author__, __version__
from visionpulse.benchmarks.evaluator import BenchmarkEvaluator
from visionpulse.config import get_config
from visionpulse.core.pipeline import VisionPipeline
from visionpulse.core.stream import StreamSource


def cmd_run(args: argparse.Namespace) -> None:
    """Run interactive vision pipeline stream."""
    cfg = get_config()
    print(f"\n========================================================")
    print(f" VisionPulse v{__version__} - Interactive Studio Stream")
    print(f" Developer: {__author__}")
    print(f" Source: {args.source} | Mode: {args.mode} | Device: {cfg.models.device}")
    print(f" Shortcuts: [m]obileNet, [y]olo, [f]usion, [s]napshot, [r]ecord, [q]uit")
    print(f"========================================================\n")

    pipeline = VisionPipeline(config=cfg, mode=args.mode)
    stream = StreamSource(source=args.source)

    if not stream.is_opened:
        print(f"Error: Unable to open stream source '{args.source}'", file=sys.stderr)
        sys.exit(1)

    pipeline.run_live(stream_source=stream, display=not args.headless)


def cmd_infer(args: argparse.Namespace) -> None:
    """Run one-off inference on a static image."""
    cfg = get_config()
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Unable to decode image file: {args.input}", file=sys.stderr)
        sys.exit(1)

    pipeline = VisionPipeline(config=cfg, mode=args.mode)
    annotated, result, telemetry = pipeline.process_frame(image, conf_threshold=args.conf)

    print("\n--- Inference Telemetry ---")
    print(f"Model: {result.model_name}")
    print(f"Mode: {result.mode}")
    print(f"Latency: {result.latency_ms:.2f} ms")
    print(f"Device: {result.device}")
    if result.context_summary:
        print(f"Summary: {result.context_summary}")

    if args.json:
        print("\n--- Structured JSON ---")
        print(result.model_dump_json(indent=2))

    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"\nAnnotated image saved to: {args.output}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Execute empirical benchmarks across models."""
    print(f"\n========================================================")
    print(f" VisionPulse Empirical Benchmark Suite")
    print(f" Trials: {args.iterations} | Warmup: {args.warmup}")
    print(f"========================================================\n")

    evaluator = BenchmarkEvaluator(iterations=args.iterations, warmup=args.warmup)
    results = evaluator.run_full_benchmark()

    print(f"{'Model / Engine':<32} | {'P50 (ms)':<9} | {'P90 (ms)':<9} | {'FPS':<7} | {'RAM (MB)':<8}")
    print("-" * 75)
    for key, data in results.items():
        print(
            f"{data['model_name']:<32} | "
            f"{data['p50_latency_ms']:<9.2f} | "
            f"{data['p90_latency_ms']:<9.2f} | "
            f"{data['throughput_fps']:<7.1f} | "
            f"{data['ram_delta_mb']:<8.2f}"
        )
    print("-" * 75)


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch FastAPI REST microservice."""
    cfg = get_config()
    host = args.host or cfg.api_host
    port = args.port or cfg.api_port
    print(f"\nStarting VisionPulse FastAPI Server on http://{host}:{port} ...")
    uvicorn.run("visionpulse.web.api:app", host=host, port=port, reload=args.reload)


def cmd_studio(args: argparse.Namespace) -> None:
    """Launch Streamlit web studio."""
    import subprocess
    app_path = os.path.join(os.path.dirname(__file__), "web", "app.py")
    port = args.port or 8501
    print(f"\nLaunching VisionPulse Streamlit Studio on port {port} ...")
    subprocess.run(["streamlit", "run", app_path, "--server.port", str(port)])


def cmd_info(args: argparse.Namespace) -> None:
    """Display system diagnostics, versions, and developer credentials."""
    import torch
    cfg = get_config()
    cuda_ok = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU (Multi-threaded)"

    print(f"\n========================================================")
    print(f" VisionPulse Perception Platform")
    print(f"========================================================")
    print(f" Version       : {__version__}")
    print(f" Developer     : {__author__}")
    print(f" Original By   : kirankumarnscr7-cmyk (mini-project4 baseline)")
    print(f" License       : MIT")
    print(f" PyTorch       : {torch.__version__}")
    print(f" Accelerator   : {dev_name}")
    print(f" CUDA Enabled  : {cuda_ok}")
    print(f" Default Model : {cfg.models.yolo_model}")
    print(f" API Host/Port : {cfg.api_host}:{cfg.api_port}")
    print(f" Studio Port   : {cfg.web_port}")
    print(f"========================================================\n")


def main() -> None:
    """CLI entrypoint dispatcher."""
    parser = argparse.ArgumentParser(
        prog="visionpulse",
        description="VisionPulse: Dual-Engine Real-Time Vision & Cognitive Object Intelligence System",
    )
    parser.add_argument("--version", action="version", version=f"VisionPulse {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 'run' subcommand
    p_run = subparsers.add_parser("run", help="Run interactive stream pipeline")
    p_run.add_argument("--source", default="0", help="Webcam index, video path, image path, or 'synthetic'")
    p_run.add_argument("--mode", choices=["detection", "classification", "fusion"], default="detection")
    p_run.add_argument("--headless", action="store_true", help="Run in headless mode without desktop window")

    # 'infer' subcommand
    p_infer = subparsers.add_parser("infer", help="Execute single-image inference")
    p_infer.add_argument("--input", "-i", required=True, help="Input image file path")
    p_infer.add_argument("--output", "-o", help="Path to save annotated output image")
    p_infer.add_argument("--mode", choices=["detection", "classification", "fusion"], default="detection")
    p_infer.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    p_infer.add_argument("--json", action="store_true", help="Print structured JSON telemetry")

    # 'benchmark' subcommand
    p_bench = subparsers.add_parser("benchmark", help="Run empirical performance benchmarks")
    p_bench.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    p_bench.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")

    # 'serve' subcommand
    p_serve = subparsers.add_parser("serve", help="Start FastAPI REST microservice")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface")
    p_serve.add_argument("--port", type=int, default=8000, help="Port")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # 'studio' subcommand
    p_studio = subparsers.add_parser("studio", help="Start Streamlit Cognitive Studio")
    p_studio.add_argument("--port", type=int, default=8501, help="Studio port")

    # 'info' subcommand
    subparsers.add_parser("info", help="Display system information and credentials")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "run": cmd_run,
        "infer": cmd_infer,
        "benchmark": cmd_benchmark,
        "serve": cmd_serve,
        "studio": cmd_studio,
        "info": cmd_info,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
