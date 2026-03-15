"""
export_model.py -- Export trained SmartSort YOLO11 weights to ONNX / TorchScript
and benchmark inference latency across formats.

Run from smartsort/scripts/:
    python export_model.py
    python export_model.py --model_path ../models/best.pt --formats onnx,torchscript
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SMARTSORT_DIR = SCRIPT_DIR.parent
LOGS_DIR = SMARTSORT_DIR / "logs"
MODELS_DIR = SMARTSORT_DIR / "models"
DEFAULT_MODEL = MODELS_DIR / "best.pt"
BENCHMARK_WARMUP = 10   # inference passes discarded before timing
BENCHMARK_RUNS = 50     # inference passes timed


# ── Logging ───────────────────────────────────────────────────────────────────────

def setup_logging(logs_dir: Path) -> logging.Logger:
    """Configure logger to write to stdout and a timestamped log file."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"export_{ts}.log"

    logger = logging.getLogger("export_model")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log file: %s", log_file)
    return logger


# ── Export ────────────────────────────────────────────────────────────────────────

def export_onnx(model_path: Path, imgsz: int, models_dir: Path, logger: logging.Logger) -> Path | None:
    """
    Export a YOLO11 .pt model to ONNX format with graph simplification.

    Args:
        model_path: Source .pt weights file.
        imgsz:      Input image size (square).
        models_dir: Destination directory.
        logger:     Logger instance.

    Returns:
        Path to exported .onnx file, or None on failure.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        logger.error("ultralytics not installed.")
        return None

    logger.info("Exporting to ONNX (imgsz=%d, simplify=True) ...", imgsz)
    try:
        model = YOLO(str(model_path))
        export_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
        # ultralytics places the export next to the .pt file; move to models_dir
        src = Path(export_path)
        dst = models_dir / "best.onnx"
        if src != dst and src.exists():
            import shutil
            shutil.copy2(src, dst)
        logger.info("ONNX exported -> %s (%.1f MB)", dst, dst.stat().st_size / 1e6)
        return dst
    except Exception as exc:  # noqa: BLE001
        logger.error("ONNX export failed: %s", exc)
        return None


def export_torchscript(model_path: Path, imgsz: int, models_dir: Path, logger: logging.Logger) -> Path | None:
    """
    Export a YOLO11 .pt model to TorchScript format.

    Args:
        model_path: Source .pt weights file.
        imgsz:      Input image size (square).
        models_dir: Destination directory.
        logger:     Logger instance.

    Returns:
        Path to exported .torchscript file, or None on failure.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        logger.error("ultralytics not installed.")
        return None

    logger.info("Exporting to TorchScript (imgsz=%d) ...", imgsz)
    try:
        model = YOLO(str(model_path))
        export_path = model.export(format="torchscript", imgsz=imgsz)
        src = Path(export_path)
        dst = models_dir / "best.torchscript"
        if src != dst and src.exists():
            import shutil
            shutil.copy2(src, dst)
        logger.info("TorchScript exported -> %s (%.1f MB)", dst, dst.stat().st_size / 1e6)
        return dst
    except Exception as exc:  # noqa: BLE001
        logger.error("TorchScript export failed: %s", exc)
        return None


# ── Benchmarking ──────────────────────────────────────────────────────────────────

def make_dummy_image(imgsz: int) -> object:
    """
    Create a random dummy numpy image for benchmarking.

    Args:
        imgsz: Height and width of the image (square).

    Returns:
        numpy array of shape (imgsz, imgsz, 3), dtype uint8.
    """
    try:
        import numpy as np  # noqa: PLC0415
        return np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    except ImportError:
        return None


def benchmark_pytorch(model_path: Path, dummy_img, warmup: int, runs: int, logger: logging.Logger) -> dict | None:
    """
    Benchmark PyTorch (.pt) model inference latency.

    Args:
        model_path: Path to .pt weights.
        dummy_img:  Numpy image for inference.
        warmup:     Number of warmup passes (discarded).
        runs:       Number of timed passes.
        logger:     Logger instance.

    Returns:
        Dict with avg_ms, fps, size_mb, or None on failure.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        return None

    if not model_path.exists():
        return None

    try:
        model = YOLO(str(model_path))
        for _ in range(warmup):
            model(dummy_img, verbose=False)

        t0 = time.perf_counter()
        for _ in range(runs):
            model(dummy_img, verbose=False)
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / runs) * 1000
        size_mb = model_path.stat().st_size / 1e6
        return {"avg_ms": avg_ms, "fps": 1000 / avg_ms, "size_mb": size_mb}
    except Exception as exc:  # noqa: BLE001
        logger.warning("PyTorch benchmark failed: %s", exc)
        return None


def benchmark_onnx(model_path: Path, dummy_img, imgsz: int, warmup: int, runs: int, logger: logging.Logger) -> dict | None:
    """
    Benchmark ONNX model inference latency using onnxruntime.

    Args:
        model_path: Path to .onnx file.
        dummy_img:  Numpy image for inference.
        imgsz:      Input image size (square).
        warmup:     Number of warmup passes (discarded).
        runs:       Number of timed passes.
        logger:     Logger instance.

    Returns:
        Dict with avg_ms, fps, size_mb, or None on failure.
    """
    try:
        import numpy as np  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError as e:
        logger.warning("ONNX benchmark skipped: %s", e)
        return None

    if not model_path.exists():
        return None

    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(str(model_path), providers=providers)
        input_name = session.get_inputs()[0].name

        # Preprocess: HWC uint8 -> NCHW float32 normalised
        img = dummy_img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))     # HWC -> CHW
        img = np.expand_dims(img, 0)            # CHW -> NCHW

        for _ in range(warmup):
            session.run(None, {input_name: img})

        t0 = time.perf_counter()
        for _ in range(runs):
            session.run(None, {input_name: img})
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / runs) * 1000
        size_mb = model_path.stat().st_size / 1e6
        return {"avg_ms": avg_ms, "fps": 1000 / avg_ms, "size_mb": size_mb}
    except Exception as exc:  # noqa: BLE001
        logger.warning("ONNX benchmark failed: %s", exc)
        return None


def benchmark_torchscript(model_path: Path, dummy_img, imgsz: int, warmup: int, runs: int, logger: logging.Logger) -> dict | None:
    """
    Benchmark TorchScript model inference latency.

    Args:
        model_path: Path to .torchscript file.
        dummy_img:  Numpy image for inference.
        imgsz:      Input image size (square).
        warmup:     Number of warmup passes (discarded).
        runs:       Number of timed passes.
        logger:     Logger instance.

    Returns:
        Dict with avg_ms, fps, size_mb, or None on failure.
    """
    try:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError as e:
        logger.warning("TorchScript benchmark skipped: %s", e)
        return None

    if not model_path.exists():
        return None

    try:
        model_ts = torch.jit.load(str(model_path))
        model_ts.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_ts = model_ts.to(device)

        img = dummy_img.astype("float32") / 255.0
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        with torch.no_grad():
            for _ in range(warmup):
                model_ts(tensor)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                model_ts(tensor)
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / runs) * 1000
        size_mb = model_path.stat().st_size / 1e6
        return {"avg_ms": avg_ms, "fps": 1000 / avg_ms, "size_mb": size_mb}
    except Exception as exc:  # noqa: BLE001
        logger.warning("TorchScript benchmark failed: %s", exc)
        return None


def print_benchmark_table(results: dict[str, dict | None], logger: logging.Logger) -> None:
    """
    Print a formatted benchmark comparison table.

    Args:
        results: {format_name: {avg_ms, fps, size_mb} or None}
        logger:  Logger instance.
    """
    logger.info("")
    logger.info("Benchmark results (%d inference passes, %d warmup):", BENCHMARK_RUNS, BENCHMARK_WARMUP)
    logger.info("%-16s %12s %10s %12s", "Format", "Avg latency", "FPS", "File size")
    logger.info("-" * 54)
    for fmt, res in results.items():
        if res is None:
            logger.info("%-16s %12s %10s %12s", fmt, "FAILED", "-", "-")
        else:
            logger.info("%-16s %11.1f ms %9.1f %10.1f MB",
                        fmt, res["avg_ms"], res["fps"], res["size_mb"])
    logger.info("-" * 54)


# ── CLI ───────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export YOLO11 model to ONNX/TorchScript and benchmark latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL,
                        help="Path to trained .pt weights file.")
    parser.add_argument("--formats", type=str, default="onnx,torchscript",
                        help="Comma-separated list of export formats.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size for export and benchmarking.")
    parser.add_argument("--no_benchmark", action="store_true",
                        help="Skip latency benchmarking (export only).")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: export model and benchmark inference latency."""
    args = parse_args()
    logger = setup_logging(LOGS_DIR)

    model_path: Path = args.model_path.resolve()
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    logger.info("Exporting %s to formats: %s", model_path.name, formats)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Exports ───────────────────────────────────────────────────────────────────
    exported: dict[str, Path | None] = {"pytorch": model_path}

    if "onnx" in formats:
        exported["onnx"] = export_onnx(model_path, args.imgsz, MODELS_DIR, logger)

    if "torchscript" in formats:
        exported["torchscript"] = export_torchscript(model_path, args.imgsz, MODELS_DIR, logger)

    # ── Benchmarking ──────────────────────────────────────────────────────────────
    if args.no_benchmark:
        logger.info("Benchmarking skipped (--no_benchmark).")
        return

    logger.info("Preparing benchmarks (warmup=%d, runs=%d) ...", BENCHMARK_WARMUP, BENCHMARK_RUNS)
    dummy = make_dummy_image(args.imgsz)
    if dummy is None:
        logger.error("numpy not installed; cannot benchmark. Run: pip install numpy")
        return

    bench_results: dict[str, dict | None] = {}

    bench_results["pytorch (.pt)"] = benchmark_pytorch(
        model_path, dummy, BENCHMARK_WARMUP, BENCHMARK_RUNS, logger)

    if exported.get("onnx"):
        bench_results["onnx"] = benchmark_onnx(
            exported["onnx"], dummy, args.imgsz, BENCHMARK_WARMUP, BENCHMARK_RUNS, logger)

    if exported.get("torchscript"):
        bench_results["torchscript"] = benchmark_torchscript(
            exported["torchscript"], dummy, args.imgsz, BENCHMARK_WARMUP, BENCHMARK_RUNS, logger)

    print_benchmark_table(bench_results, logger)
    logger.info("Export complete.")


if __name__ == "__main__":
    main()
