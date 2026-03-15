"""
download_roboflow.py — Bulk Roboflow dataset downloader for SmartSort.

Run from smartsort/scripts/:
    python download_roboflow.py --api_key YOUR_KEY
    python download_roboflow.py --api_key YOUR_KEY --dry_run

Datasets downloaded (in YOLOv8 / YOLO11 format):
  PRIORITY 1 — trash-detection-main   (2783 images, 64 classes)
  PRIORITY 2 — food-waste             (7622 images, 32 food classes)
  PRIORITY 3 — trash-detections-fyp   (recyclable variety)
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # smartsort/scripts/
SMARTSORT_DIR = SCRIPT_DIR.parent                     # smartsort/
LOGS_DIR = SMARTSORT_DIR / "logs"
DEFAULT_OUTPUT_DIR = SMARTSORT_DIR.parent / "roboflow_raw"

# ── Dataset registry ─────────────────────────────────────────────────────────────
DATASETS = [
    {
        "priority": 1,
        "label": "trash-detection-main",
        "workspace": "trash-dataset-for-oriented-bounded-box",
        "project": "trash-detection-1fjjc",
        "version": 14,
        "dest_dir": "trash-detection-main",
        "note": "2783 images, 64 classes (Battery, Cigarette, Syringe, Food waste, …)",
    },
    {
        "priority": 2,
        "label": "food-waste",
        "workspace": "abrars-models",
        "project": "food-waste-detection-yolo-v8",
        "version": 1,
        "dest_dir": "food-waste",
        "note": "7622 images, 32 food classes — boosts Wet category coverage",
    },
    {
        "priority": 3,
        "label": "trash-detections-fyp",
        "workspace": "fyp-bfx3h",
        "project": "yolov8-trash-detections",
        "version": 1,
        "dest_dir": "trash-detections-fyp",
        "note": "Recyclable variety — boosts Dry category coverage",
    },
]


# ── Logging setup ─────────────────────────────────────────────────────────────────
def setup_logging(logs_dir: Path) -> logging.Logger:
    """Configure root logger to write to both stdout and a timestamped log file."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"download_{timestamp}.log"

    logger = logging.getLogger("download_roboflow")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log file: %s", log_file)
    return logger


# ── Dry-run helper ────────────────────────────────────────────────────────────────
def print_dry_run(datasets: list[dict], output_dir: Path, logger: logging.Logger) -> None:
    """Print what would be downloaded without touching the network."""
    logger.info("=== DRY RUN — nothing will be downloaded ===")
    for ds in datasets:
        logger.info(
            "[P%d] %s  →  %s",
            ds["priority"],
            ds["label"],
            output_dir / ds["dest_dir"],
        )
        logger.info("      workspace : %s", ds["workspace"])
        logger.info("      project   : %s", ds["project"])
        logger.info("      version   : %s", ds["version"])
        logger.info("      note      : %s", ds["note"])


# ── Single dataset downloader ─────────────────────────────────────────────────────
def download_dataset(
    rf,
    ds: dict,
    output_dir: Path,
    logger: logging.Logger,
) -> bool:
    """
    Download one Roboflow dataset in YOLOv8 format.

    Args:
        rf:         Authenticated roboflow.Roboflow instance.
        ds:         Dataset metadata dict from DATASETS registry.
        output_dir: Root roboflow_raw/ directory.
        logger:     Logger instance.

    Returns:
        True on success, False on any error.
    """
    label = ds["label"]
    dest = output_dir / ds["dest_dir"]
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("─" * 60)
    logger.info("[P%d] Downloading: %s", ds["priority"], label)
    logger.info("      %s", ds["note"])
    logger.info("      destination : %s", dest)

    try:
        workspace = rf.workspace(ds["workspace"])
        project = workspace.project(ds["project"])
        version = project.version(ds["version"])

        # Download into dest — Roboflow SDK creates a subfolder inside dest
        dataset = version.download("yolov8", location=str(dest), overwrite=True)

        # Report what was found
        try:
            classes = dataset.classes if hasattr(dataset, "classes") else []
            img_count = (
                len(list(dest.rglob("*.jpg"))) + len(list(dest.rglob("*.png")))
            )
            logger.info("      classes found : %d", len(classes))
            logger.info("      images found  : %d", img_count)
            if classes:
                logger.debug("      class list: %s", classes)
        except Exception as info_err:  # noqa: BLE001
            logger.debug("Could not read dataset metadata: %s", info_err)

        logger.info("[P%d] ✓ %s downloaded successfully", ds["priority"], label)
        return True

    except Exception as exc:  # noqa: BLE001
        logger.error("[P%d] ✗ Failed to download %s: %s", ds["priority"], label, exc)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bulk-download Roboflow datasets for SmartSort in YOLOv8 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api_key",
        required=True,
        help="Roboflow API key (find it in roboflow.com → Account Settings).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory where downloaded datasets are saved.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Download only datasets at or above this priority level (1 = highest).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: download all configured Roboflow datasets."""
    args = parse_args()
    logger = setup_logging(LOGS_DIR)

    output_dir: Path = args.output_dir.resolve()
    logger.info("Output directory : %s", output_dir)

    # Filter by priority if requested
    datasets = DATASETS
    if args.priority is not None:
        datasets = [ds for ds in DATASETS if ds["priority"] <= args.priority]
        logger.info("Filtering to priority ≤ %d (%d datasets)", args.priority, len(datasets))

    if args.dry_run:
        print_dry_run(datasets, output_dir, logger)
        return

    # Authenticate with Roboflow
    try:
        from roboflow import Roboflow  # noqa: PLC0415
    except ImportError:
        logger.error("roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    logger.info("Authenticating with Roboflow …")
    try:
        rf = Roboflow(api_key=args.api_key)
    except Exception as exc:  # noqa: BLE001
        logger.error("Authentication failed: %s", exc)
        sys.exit(1)

    # Download each dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for ds in datasets:
        ok = download_dataset(rf, ds, output_dir, logger)
        results[ds["label"]] = ok

    # Summary
    logger.info("=" * 60)
    logger.info("Download summary:")
    for label, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED"
        logger.info("  %s  %s", status, label)

    failed = [k for k, v in results.items() if not v]
    if failed:
        logger.warning("%d dataset(s) failed. Check the log for details.", len(failed))
        sys.exit(1)
    else:
        logger.info("All datasets downloaded successfully.")


if __name__ == "__main__":
    main()
