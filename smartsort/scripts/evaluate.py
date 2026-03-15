"""
evaluate.py -- Comprehensive SmartSort model evaluation.

Runs per-class metrics, generates confusion matrix and confidence histograms,
saves annotated sample predictions, and writes a Markdown evaluation report.

Run from smartsort/scripts/:
    python evaluate.py
    python evaluate.py --model_path ../models/best.onnx
    python evaluate.py --conf_threshold 0.35 --save_samples 30
"""

import argparse
import json
import logging
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SMARTSORT_DIR = SCRIPT_DIR.parent
LOGS_DIR = SMARTSORT_DIR / "logs"
MODELS_DIR = SMARTSORT_DIR / "models"
DEFAULT_MODEL = MODELS_DIR / "best.pt"
DEFAULT_DATA = SMARTSORT_DIR.parent / "merged_dataset" / "dataset.yaml"

TARGET_NAMES = ["Recyclable", "Organic", "General Waste"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Logging ───────────────────────────────────────────────────────────────────────

def setup_logging(logs_dir: Path) -> logging.Logger:
    """Configure logger to write to stdout and a timestamped log file."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"evaluate_{ts}.log"

    logger = logging.getLogger("evaluate")
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


# ── Data helpers ──────────────────────────────────────────────────────────────────

def load_dataset_yaml(data_yaml: Path, logger: logging.Logger) -> dict:
    """
    Load and parse a YOLO dataset.yaml file.

    Args:
        data_yaml: Path to dataset.yaml.
        logger:    Logger instance.

    Returns:
        Parsed dict, or empty dict on failure.
    """
    try:
        import yaml  # noqa: PLC0415
        with open(data_yaml, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot load dataset yaml: %s", exc)
        return {}


def get_test_images(data_yaml: Path, logger: logging.Logger) -> list[Path]:
    """
    Return a list of all image paths in the test split of the dataset.

    Args:
        data_yaml: Path to dataset.yaml.
        logger:    Logger instance.

    Returns:
        Sorted list of image Paths.
    """
    ds = load_dataset_yaml(data_yaml, logger)
    if not ds:
        return []

    root = Path(ds.get("path", data_yaml.parent))
    test_rel = ds.get("test", "images/test")
    test_dir = root / test_rel

    if not test_dir.exists():
        logger.warning("Test images dir not found: %s", test_dir)
        return []

    images = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    logger.info("Found %d test images in %s", len(images), test_dir)
    return images


def count_test_boxes(data_yaml: Path, nc: int, logger: logging.Logger) -> dict[str, int]:
    """
    Count ground-truth bounding boxes per class in the test split.

    Args:
        data_yaml: Path to dataset.yaml.
        nc:        Number of classes.
        logger:    Logger instance.

    Returns:
        {class_name: count}
    """
    ds = load_dataset_yaml(data_yaml, logger)
    if not ds:
        return {}

    root = Path(ds.get("path", data_yaml.parent))
    test_rel = ds.get("test", "images/test")
    lbl_dir = root / test_rel.replace("images", "labels")

    counts = [0] * nc
    if lbl_dir.exists():
        for lf in lbl_dir.glob("*.txt"):
            try:
                for line in lf.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if parts:
                        try:
                            cls = int(parts[0])
                            if 0 <= cls < nc:
                                counts[cls] += 1
                        except ValueError:
                            pass
            except OSError:
                pass

    return {TARGET_NAMES[i]: counts[i] for i in range(nc)}


# ── Core evaluation ───────────────────────────────────────────────────────────────

def run_val(
    model_path: Path,
    data_yaml: Path,
    conf: float,
    logger: logging.Logger,
) -> tuple[object | None, dict]:
    """
    Run model.val() on the test split and return raw results + parsed metrics dict.

    Args:
        model_path: Path to .pt (or .onnx) weights.
        data_yaml:  Path to dataset.yaml.
        conf:       Confidence threshold.
        logger:     Logger instance.

    Returns:
        (val_results_object, metrics_dict)
        metrics_dict has keys: per_class, mAP50, mAP50_95, precision_mean, recall_mean
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        logger.error("ultralytics not installed.")
        return None, {}

    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        return None, {}

    logger.info("Running val() on test split with conf=%.2f ...", conf)
    model = YOLO(str(model_path))

    try:
        results = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=640,
            conf=conf,
            iou=0.6,
            verbose=True,
            save_json=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("model.val() failed: %s", exc)
        return None, {}

    metrics: dict = {}
    try:
        mp = float(results.box.mp)
        mr = float(results.box.mr)
        map50 = float(results.box.map50)
        map5095 = float(results.box.map)

        metrics["mAP50"] = round(map50, 4)
        metrics["mAP50_95"] = round(map5095, 4)
        metrics["precision_mean"] = round(mp, 4)
        metrics["recall_mean"] = round(mr, 4)

        per_class = []
        for i, name in enumerate(TARGET_NAMES):
            try:
                p = float(results.box.p[i])
                r = float(results.box.r[i])
                ap50 = float(results.box.ap50[i])
                ap = float(results.box.ap[i])
                f1 = 2 * p * r / (p + r + 1e-9)
            except (IndexError, TypeError, AttributeError):
                p = r = ap50 = ap = f1 = 0.0

            per_class.append({
                "class": name,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "mAP50": round(ap50, 4),
                "mAP50_95": round(ap, 4),
            })

        metrics["per_class"] = per_class

    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse metrics from val results: %s", exc)

    return results, metrics


def print_metrics_table(metrics: dict, box_counts: dict[str, int], logger: logging.Logger) -> None:
    """
    Print a formatted per-class metrics table to the log.

    Args:
        metrics:    Parsed metrics dict from run_val().
        box_counts: {class_name: gt_box_count} from count_test_boxes().
        logger:     Logger instance.
    """
    logger.info("")
    logger.info("%-16s %10s %8s %6s %8s %10s %8s",
                "Class", "Precision", "Recall", "F1", "mAP@50", "mAP@50-95", "# Boxes")
    logger.info("-" * 74)

    for row in metrics.get("per_class", []):
        n_boxes = box_counts.get(row["class"], 0)
        logger.info("%-16s %10.3f %8.3f %6.3f %8.3f %10.3f %8d",
                    row["class"],
                    row["precision"], row["recall"], row["f1"],
                    row["mAP50"], row["mAP50_95"], n_boxes)

    logger.info("-" * 74)
    logger.info("%-16s %10.3f %8.3f %6s %8.3f %10.3f",
                "MEAN",
                metrics.get("precision_mean", 0),
                metrics.get("recall_mean", 0),
                "-",
                metrics.get("mAP50", 0),
                metrics.get("mAP50_95", 0))


# ── Visualisations ────────────────────────────────────────────────────────────────

def copy_confusion_matrix(val_results, logs_dir: Path, ts: str, logger: logging.Logger) -> Path | None:
    """
    Copy the YOLO-generated confusion matrix PNG to logs/.

    Args:
        val_results: ultralytics val results object.
        logs_dir:    Destination logs directory.
        ts:          Timestamp string for the filename.
        logger:      Logger instance.

    Returns:
        Path to the copied PNG, or None if not found.
    """
    if val_results is None:
        return None
    try:
        save_dir = Path(val_results.save_dir)
        candidates = list(save_dir.glob("confusion_matrix*.png"))
        if not candidates:
            logger.warning("No confusion matrix PNG found in %s", save_dir)
            return None
        dst = logs_dir / f"confusion_matrix_{ts}.png"
        shutil.copy2(candidates[0], dst)
        logger.info("Confusion matrix saved -> %s", dst)
        return dst
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not copy confusion matrix: %s", exc)
        return None


def plot_confidence_histograms(
    model_path: Path,
    test_images: list[Path],
    logs_dir: Path,
    ts: str,
    conf: float,
    logger: logging.Logger,
) -> Path | None:
    """
    Run inference on up to 500 test images and plot per-class confidence distributions.

    Args:
        model_path:   Path to model weights.
        test_images:  List of test image paths.
        logs_dir:     Output directory.
        ts:           Timestamp string for the filename.
        conf:         Confidence threshold for inference.
        logger:       Logger instance.

    Returns:
        Path to saved histogram PNG, or None on failure.
    """
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as e:
        logger.warning("Skipping confidence histograms: %s", e)
        return None

    if not model_path.exists() or not test_images:
        return None

    logger.info("Generating confidence histograms (sampling up to 500 images) ...")
    model = YOLO(str(model_path))

    sample = random.sample(test_images, min(500, len(test_images)))
    class_confs: dict[int, list[float]] = {i: [] for i in range(len(TARGET_NAMES))}

    try:
        for img_path in sample:
            preds = model(str(img_path), conf=conf, verbose=False)
            for r in preds:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls = int(box.cls.item())
                    c = float(box.conf.item())
                    if cls in class_confs:
                        class_confs[cls].append(c)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Inference for confidence histogram failed: %s", exc)
        return None

    nc = len(TARGET_NAMES)
    fig, axes = plt.subplots(1, nc, figsize=(5 * nc, 4), sharey=True)
    colors = ["steelblue", "seagreen", "tomato"]

    for i, ax in enumerate(axes):
        confs = class_confs[i]
        if confs:
            ax.hist(confs, bins=20, range=(0, 1), color=colors[i], edgecolor="white", alpha=0.85)
        ax.set_title(TARGET_NAMES[i], fontsize=12)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count" if i == 0 else "")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Predicted Confidence Distribution by Class (test set)", fontsize=13)
    fig.tight_layout()

    out_path = logs_dir / f"confidence_hist_{ts}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Confidence histograms saved -> %s", out_path)
    return out_path


def save_sample_predictions(
    model_path: Path,
    test_images: list[Path],
    n_samples: int,
    logs_dir: Path,
    ts: str,
    conf: float,
    logger: logging.Logger,
) -> Path | None:
    """
    Run inference on n_samples random test images and save annotated results.

    Args:
        model_path:  Path to model weights.
        test_images: Full list of test image paths to sample from.
        n_samples:   Number of images to annotate.
        logs_dir:    Parent logs directory (samples saved in a subdirectory).
        ts:          Timestamp string for the subdirectory name.
        conf:        Confidence threshold.
        logger:      Logger instance.

    Returns:
        Path to the sample_predictions directory, or None on failure.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        logger.warning("ultralytics not installed; skipping sample predictions.")
        return None

    if not model_path.exists() or not test_images:
        return None

    samples_dir = logs_dir / f"sample_predictions_{ts}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    sample = random.sample(test_images, min(n_samples, len(test_images)))
    logger.info("Saving %d annotated sample predictions -> %s", len(sample), samples_dir)

    model = YOLO(str(model_path))

    for img_path in sample:
        try:
            results = model(str(img_path), conf=conf, verbose=False)
            for r in results:
                annotated = r.plot()  # returns BGR numpy array
                out_name = samples_dir / img_path.name
                # Save using PIL (avoids cv2 dependency)
                try:
                    from PIL import Image  # noqa: PLC0415
                    import numpy as np  # noqa: PLC0415
                    # r.plot() returns BGR; convert to RGB for PIL
                    rgb = annotated[:, :, ::-1]
                    Image.fromarray(rgb).save(str(out_name))
                except ImportError:
                    # Fall back to cv2 if available
                    try:
                        import cv2  # noqa: PLC0415
                        cv2.imwrite(str(out_name), annotated)
                    except ImportError:
                        logger.warning("Neither Pillow nor cv2 available; cannot save sample images.")
                        return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prediction failed for %s: %s", img_path.name, exc)

    logger.info("Sample predictions saved -> %s", samples_dir)
    return samples_dir


# ── Markdown report ───────────────────────────────────────────────────────────────

def write_markdown_report(
    model_path: Path,
    metrics: dict,
    box_counts: dict[str, int],
    logs_dir: Path,
    ts: str,
    confusion_matrix_path: Path | None,
    hist_path: Path | None,
    logger: logging.Logger,
) -> Path:
    """
    Generate a Markdown evaluation report and save it to logs/.

    Args:
        model_path:            Evaluated model path.
        metrics:               Parsed metrics dict from run_val().
        box_counts:            GT box counts per class in test set.
        logs_dir:              Output logs directory.
        ts:                    Timestamp string.
        confusion_matrix_path: Path to confusion matrix image, or None.
        hist_path:             Path to confidence histogram image, or None.
        logger:                Logger instance.

    Returns:
        Path to the written .md report file.
    """
    model_size_mb = model_path.stat().st_size / 1e6 if model_path.exists() else 0.0
    report_path = logs_dir / f"eval_report_{ts}.md"

    weakest = None
    worst_map = 1.0
    for row in metrics.get("per_class", []):
        if row["mAP50"] < worst_map:
            worst_map = row["mAP50"]
            weakest = row["class"]

    lines = [
        f"# SmartSort Evaluation Report",
        f"",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Model:** `{model_path.name}` ({model_size_mb:.1f} MB)  ",
        f"**Model path:** `{model_path}`  ",
        f"",
        f"## Overall Metrics",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| mAP@50 | {metrics.get('mAP50', 0):.4f} |",
        f"| mAP@50-95 | {metrics.get('mAP50_95', 0):.4f} |",
        f"| Mean Precision | {metrics.get('precision_mean', 0):.4f} |",
        f"| Mean Recall | {metrics.get('recall_mean', 0):.4f} |",
        f"",
        f"## Per-Class Metrics",
        f"",
        f"| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 | GT Boxes |",
        f"|---|---|---|---|---|---|---|",
    ]

    for row in metrics.get("per_class", []):
        n_boxes = box_counts.get(row["class"], 0)
        lines.append(
            f"| {row['class']} "
            f"| {row['precision']:.3f} "
            f"| {row['recall']:.3f} "
            f"| {row['f1']:.3f} "
            f"| {row['mAP50']:.3f} "
            f"| {row['mAP50_95']:.3f} "
            f"| {n_boxes} |"
        )

    lines += [
        f"",
        f"## Class Distribution in Test Set",
        f"",
        f"| Class | GT Boxes |",
        f"|---|---|",
    ]
    for name, cnt in box_counts.items():
        lines.append(f"| {name} | {cnt} |")

    if weakest:
        lines += [
            f"",
            f"## Notes",
            f"",
            f"- **Weakest class:** `{weakest}` with mAP@50 = {worst_map:.3f}",
            f"  Consider collecting more training data or adding targeted augmentation for this class.",
        ]

    if confusion_matrix_path:
        rel_cm = confusion_matrix_path.name
        lines += [f"", f"## Confusion Matrix", f"", f"![Confusion Matrix]({rel_cm})"]

    if hist_path:
        rel_hist = hist_path.name
        lines += [f"", f"## Confidence Distributions", f"", f"![Confidence Histogram]({rel_hist})"]

    lines += [f"", f"---", f"*Generated by SmartSort evaluate.py*", f""]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Evaluation report saved -> %s", report_path)
    return report_path


# ── CLI ───────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive SmartSort model evaluation with per-class metrics, "
                    "confusion matrix, confidence histograms, and Markdown report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL,
                        help="Path to .pt or .onnx model weights.")
    parser.add_argument("--data_yaml", type=Path, default=DEFAULT_DATA,
                        help="Path to dataset.yaml.")
    parser.add_argument("--conf_threshold", type=float, default=0.25,
                        help="Confidence threshold for inference.")
    parser.add_argument("--save_samples", type=int, default=20,
                        help="Number of random test images to annotate and save.")
    parser.add_argument("--no_histograms", action="store_true",
                        help="Skip confidence histogram generation.")
    parser.add_argument("--no_samples", action="store_true",
                        help="Skip sample prediction images.")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: run full model evaluation pipeline."""
    args = parse_args()
    logger = setup_logging(LOGS_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path: Path = args.model_path.resolve()
    data_yaml: Path = args.data_yaml.resolve()

    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)
    if not data_yaml.exists():
        logger.error("dataset.yaml not found: %s", data_yaml)
        sys.exit(1)

    logger.info("Model:       %s (%.1f MB)", model_path, model_path.stat().st_size / 1e6)
    logger.info("Data yaml:   %s", data_yaml)
    logger.info("Conf thresh: %.2f", args.conf_threshold)

    # ── Evaluate ──────────────────────────────────────────────────────────────────
    val_results, metrics = run_val(model_path, data_yaml, args.conf_threshold, logger)

    # GT box counts for the test split
    box_counts = count_test_boxes(data_yaml, len(TARGET_NAMES), logger)

    # Print metrics table
    print_metrics_table(metrics, box_counts, logger)

    # ── Confusion matrix ──────────────────────────────────────────────────────────
    cm_path = copy_confusion_matrix(val_results, LOGS_DIR, ts, logger)

    # ── Test images list (needed for histograms + samples) ─────────────────────
    test_images = get_test_images(data_yaml, logger)

    # ── Confidence histograms ─────────────────────────────────────────────────────
    hist_path = None
    if not args.no_histograms:
        hist_path = plot_confidence_histograms(
            model_path, test_images, LOGS_DIR, ts, args.conf_threshold, logger)

    # ── Sample predictions ────────────────────────────────────────────────────────
    if not args.no_samples and args.save_samples > 0:
        save_sample_predictions(
            model_path, test_images, args.save_samples, LOGS_DIR, ts, args.conf_threshold, logger)

    # ── Save metrics JSON ─────────────────────────────────────────────────────────
    metrics["model"] = str(model_path)
    metrics["timestamp"] = ts
    metrics["gt_box_counts"] = box_counts
    json_path = LOGS_DIR / f"eval_{ts}.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Metrics JSON saved -> %s", json_path)

    # ── Markdown report ───────────────────────────────────────────────────────────
    write_markdown_report(
        model_path, metrics, box_counts, LOGS_DIR, ts, cm_path, hist_path, logger)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
