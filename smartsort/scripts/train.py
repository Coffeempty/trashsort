"""
train.py -- SmartSort YOLO11 training script.

Loads hyperparameters from configs/train_config.yaml, computes class weights,
optionally initialises W&B, trains the model, and auto-evaluates on the test set.

Run from smartsort/scripts/:
    python train.py
    python train.py --epochs 200 --batch 4
    python train.py --eval_only
    python train.py --resume
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SMARTSORT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SMARTSORT_DIR.parent           # trashsort/
LOGS_DIR = SMARTSORT_DIR / "logs"
MODELS_DIR = SMARTSORT_DIR / "models"
DEFAULT_CONFIG = SMARTSORT_DIR / "configs" / "train_config.yaml"

TARGET_NAMES = ["Recyclable", "Organic", "General Waste"]
MAX_CLASS_WEIGHT = 5.0


# ── Logging ───────────────────────────────────────────────────────────────────────

def setup_logging(logs_dir: Path) -> logging.Logger:
    """Configure root logger to write to stdout and a timestamped log file."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"train_{ts}.log"

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log file: %s", log_file)
    return logger


# ── Config loading ────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """
    Load train_config.yaml and return as a plain dict.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dict of all config keys/values.
    """
    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml not installed. Run: pip install pyyaml")
        sys.exit(1)

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def apply_cli_overrides(cfg: dict, overrides: dict) -> dict:
    """
    Apply CLI argument overrides to the config dict.

    Performs type coercion: if the config already has a value for the key,
    the override string is cast to match that type. Unknown keys are added as-is
    (string), letting YOLO handle them.

    Args:
        cfg:       Base config dict loaded from YAML.
        overrides: {key: value_str} dict from argparse unknowns.

    Returns:
        Updated config dict.
    """
    for key, val_str in overrides.items():
        if key in cfg and cfg[key] is not None:
            orig_type = type(cfg[key])
            try:
                if orig_type is bool:
                    cfg[key] = val_str.lower() in ("1", "true", "yes")
                else:
                    cfg[key] = orig_type(val_str)
            except (ValueError, TypeError):
                cfg[key] = val_str
        else:
            # Try to auto-parse common types
            if val_str.lower() in ("true", "false"):
                cfg[key] = val_str.lower() == "true"
            else:
                try:
                    cfg[key] = int(val_str)
                except ValueError:
                    try:
                        cfg[key] = float(val_str)
                    except ValueError:
                        cfg[key] = val_str
    return cfg


# ── Class weight computation ──────────────────────────────────────────────────────

def compute_class_weights(
    labels_dir: Path,
    nc: int,
    logger: logging.Logger,
) -> list[float]:
    """
    Scan all label .txt files in labels_dir and compute inverse-frequency class weights.

    Weight formula:
        raw_weight_i = max_count / count_i
        weight_i = min(raw_weight_i, MAX_CLASS_WEIGHT)

    Args:
        labels_dir: Path to merged_dataset/labels/train/.
        nc:         Number of target classes.
        logger:     Logger instance.

    Returns:
        List of floats, one weight per class (length == nc).
    """
    counts = [0] * nc

    if not labels_dir.exists():
        logger.warning("Labels dir not found: %s -- using uniform weights.", labels_dir)
        return [1.0] * nc

    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        logger.warning("No label files found in %s -- using uniform weights.", labels_dir)
        return [1.0] * nc

    for lf in label_files:
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

    logger.info("Box counts per class (train): %s",
                {TARGET_NAMES[i]: counts[i] for i in range(nc)})

    max_count = max(counts) if any(c > 0 for c in counts) else 1
    weights = []
    for i, cnt in enumerate(counts):
        if cnt == 0:
            logger.warning("Class %d (%s) has 0 boxes -- assigning max weight %.1f",
                           i, TARGET_NAMES[i], MAX_CLASS_WEIGHT)
            weights.append(MAX_CLASS_WEIGHT)
        else:
            w = min(max_count / cnt, MAX_CLASS_WEIGHT)
            weights.append(round(w, 4))

    weight_str = ", ".join(f"{TARGET_NAMES[i]}={weights[i]:.4f}" for i in range(nc))
    logger.info("Class weights: %s", weight_str)
    return weights


def count_dataset_stats(data_yaml_path: Path, logger: logging.Logger) -> dict:
    """
    Count images and bounding boxes per split from a YOLO dataset.yaml.

    Args:
        data_yaml_path: Path to dataset.yaml.
        logger:         Logger instance.

    Returns:
        Dict with keys like "train_images", "val_boxes", etc.
    """
    try:
        import yaml
    except ImportError:
        return {}

    stats: dict = {}
    try:
        with open(data_yaml_path, encoding="utf-8") as f:
            ds = yaml.safe_load(f)
        dataset_root = Path(ds.get("path", data_yaml_path.parent))
        nc = ds.get("nc", 3)

        for split in ("train", "val", "test"):
            img_rel = ds.get(split, f"images/{split}")
            img_dir = dataset_root / img_rel
            lbl_dir = dataset_root / img_rel.replace("images", "labels")

            if img_dir.exists():
                img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                n_images = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in img_exts)
                stats[f"{split}_images"] = n_images

            if lbl_dir.exists():
                box_counts = [0] * nc
                for lf in lbl_dir.glob("*.txt"):
                    try:
                        for line in lf.read_text(encoding="utf-8").splitlines():
                            parts = line.strip().split()
                            if parts:
                                try:
                                    cls = int(parts[0])
                                    if 0 <= cls < nc:
                                        box_counts[cls] += 1
                                except ValueError:
                                    pass
                    except OSError:
                        pass
                stats[f"{split}_boxes"] = {TARGET_NAMES[i]: box_counts[i] for i in range(nc)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not compute dataset stats: %s", exc)

    return stats


# ── W&B integration ───────────────────────────────────────────────────────────────

def init_wandb(
    cfg: dict,
    class_weights: list[float],
    dataset_stats: dict,
    logger: logging.Logger,
) -> object | None:
    """
    Initialise a Weights & Biases run with all hyperparameters and dataset stats.

    Gracefully returns None on any failure (not logged in, no network, etc.).

    Args:
        cfg:            Full config dict (hyperparams).
        class_weights:  Computed per-class weights.
        dataset_stats:  Dict from count_dataset_stats().
        logger:         Logger instance.

    Returns:
        Active wandb.Run object, or None if W&B is unavailable.
    """
    if not cfg.get("wandb_enabled", True):
        logger.info("W&B disabled in config (wandb_enabled: false).")
        return None

    try:
        import wandb  # noqa: PLC0415

        # Flatten config for W&B (remove W&B-specific keys to avoid confusion)
        log_cfg = {k: v for k, v in cfg.items() if not k.startswith("wandb_")}
        log_cfg["class_weights"] = {TARGET_NAMES[i]: class_weights[i] for i in range(len(class_weights))}
        log_cfg.update(dataset_stats)

        run = wandb.init(
            project=cfg.get("wandb_project", "smartsort"),
            config=log_cfg,
            resume="allow",
            name=cfg.get("name", "train"),
        )
        logger.info("W&B run initialised: %s", run.url if run else "unknown")
        return run

    except Exception as exc:  # noqa: BLE001
        logger.warning("W&B unavailable, continuing with local logging only. Reason: %s", exc)
        return None


def log_metrics_to_wandb(run, metrics: dict, logger: logging.Logger) -> None:
    """
    Log a flat metrics dict to an active W&B run.

    Args:
        run:     Active wandb.Run, or None.
        metrics: Dict of metric name -> value.
        logger:  Logger instance.
    """
    if run is None:
        return
    try:
        run.log(metrics)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to log metrics to W&B: %s", exc)


def finish_wandb(run, logger: logging.Logger) -> None:
    """
    Cleanly finish a W&B run.

    Args:
        run:    Active wandb.Run, or None.
        logger: Logger instance.
    """
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:  # noqa: BLE001
        logger.warning("W&B finish() failed: %s", exc)


# ── Training ──────────────────────────────────────────────────────────────────────

# Whitelist of kwargs accepted by model.train() in YOLO11 / ultralytics.
# NOTE: cls_pw / obj_pw are NOT valid in modern ultralytics — use cls= (loss gain)
# to indirectly weight minority classes.  YOLO11 does not support per-class
# loss weighting natively; we print the computed weights for information only.
VALID_YOLO_ARGS = {
    "data", "epochs", "imgsz", "batch", "patience",
    "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum",
    "augment", "mosaic", "mixup", "copy_paste",
    "degrees", "translate", "scale", "fliplr", "flipud",
    "hsv_h", "hsv_s", "hsv_v", "erasing",
    "workers", "device", "project", "name", "exist_ok",
    "cls", "box", "dfl",
    "save", "save_period", "val", "plots", "resume", "amp",
    "fraction", "cache", "rect", "cos_lr", "seed", "deterministic",
    "single_cls", "image_weights", "multi_scale", "overlap_mask",
    "mask_ratio", "nbs", "close_mosaic", "dropout", "verbose",
    "optimizer", "freeze",
}


def build_train_kwargs(cfg: dict, logger: logging.Logger) -> dict:
    """
    Extract valid model.train() keyword arguments from the full config dict.

    Filters to VALID_YOLO_ARGS only; logs any config keys that are removed so
    the user knows they were not passed to YOLO.

    Args:
        cfg:    Full config dict (may contain non-YOLO keys like wandb_*).
        logger: Logger instance.

    Returns:
        Dict of kwargs ready to pass to model.train(**kwargs).
    """
    kwargs = {}
    removed = []
    for k, v in cfg.items():
        if v is None:
            continue
        if k in VALID_YOLO_ARGS:
            kwargs[k] = v
        else:
            removed.append(k)

    if removed:
        logger.warning("Config keys not passed to YOLO (not valid YOLO args): %s", removed)

    return kwargs


def run_training(cfg: dict, class_weights: list[float], logger: logging.Logger) -> Path | None:
    """
    Execute YOLO11 training with OOM fallback.

    Attempts training at the configured batch size. On CUDA OOM, halves the
    batch size and retries once. Exits gracefully if still OOM.

    Args:
        cfg:           Full config dict.
        class_weights: Computed per-class weights (printed for information only;
                       YOLO11 does not support per-class loss weighting natively).
        logger:        Logger instance.

    Returns:
        Path to the run directory (containing weights/best.pt), or None on failure.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model_path = cfg.get("model", "yolo11m.pt")
    logger.info("Loading model: %s", model_path)
    model = YOLO(model_path)

    kwargs = build_train_kwargs(cfg, logger)
    logger.info("Training kwargs: %s", kwargs)

    def _attempt(batch: int) -> object:
        """Run model.train() with a given batch size."""
        kwargs["batch"] = batch
        logger.info("Starting training (batch=%d, epochs=%d) ...", batch, kwargs.get("epochs", 150))
        return model.train(**kwargs)

    try:
        results = _attempt(kwargs.get("batch", 8))
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            fallback_batch = max(2, kwargs.get("batch", 8) // 2)
            logger.warning("CUDA OOM! Retrying with batch=%d ...", fallback_batch)
            try:
                results = _attempt(fallback_batch)
            except RuntimeError as exc2:
                if "out of memory" in str(exc2).lower():
                    logger.error(
                        "Still OOM with batch=%d. "
                        "Try: python train.py --batch 2 --imgsz 512",
                        fallback_batch,
                    )
                    sys.exit(1)
                raise
        else:
            raise

    # Determine the run save directory
    try:
        save_dir = Path(results.save_dir)
    except AttributeError:
        # Fallback: find the most recently modified run directory
        runs_root = Path(cfg.get("project", SMARTSORT_DIR / "runs")) / "detect"
        if runs_root.exists():
            dirs = sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            save_dir = dirs[0] if dirs else None
        else:
            save_dir = None

    return save_dir


def copy_weights(run_dir: Path | None, models_dir: Path, logger: logging.Logger) -> Path | None:
    """
    Copy best.pt and last.pt from the YOLO run directory to smartsort/models/.

    Args:
        run_dir:    YOLO run save directory (contains weights/ subdirectory).
        models_dir: Destination directory (smartsort/models/).
        logger:     Logger instance.

    Returns:
        Path to the copied best.pt, or None if not found.
    """
    if run_dir is None:
        logger.warning("No run directory found; skipping weight copy.")
        return None

    models_dir.mkdir(parents=True, exist_ok=True)
    best_src = run_dir / "weights" / "best.pt"
    last_src = run_dir / "weights" / "last.pt"

    best_dst = None
    if best_src.exists():
        best_dst = models_dir / "best.pt"
        shutil.copy2(best_src, best_dst)
        logger.info("Saved best.pt -> %s", best_dst)
    else:
        logger.warning("best.pt not found at %s", best_src)

    if last_src.exists():
        last_dst = models_dir / "last.pt"
        shutil.copy2(last_src, last_dst)
        logger.info("Saved last.pt -> %s", last_dst)

    return best_dst


# ── Post-training evaluation ──────────────────────────────────────────────────────

def run_evaluation(
    model_path: Path,
    data_yaml: str,
    cfg: dict,
    logger: logging.Logger,
) -> dict:
    """
    Evaluate a trained model on the TEST split and return a metrics dict.

    Args:
        model_path: Path to best.pt.
        data_yaml:  Path to dataset.yaml (string, as YOLO expects).
        cfg:        Training config (used for imgsz, device, etc.).
        logger:     Logger instance.

    Returns:
        Dict of evaluation metrics.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        logger.error("ultralytics not installed.")
        return {}

    if not model_path.exists():
        logger.error("Model not found for evaluation: %s", model_path)
        return {}

    logger.info("Evaluating %s on test split ...", model_path)
    model = YOLO(str(model_path))

    try:
        val_results = model.val(
            data=data_yaml,
            split="test",
            imgsz=cfg.get("imgsz", 640),
            device=cfg.get("device", 0),
            conf=0.25,
            iou=0.6,
            verbose=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Evaluation failed: %s", exc)
        return {}

    metrics: dict = {}
    try:
        # Overall metrics
        mp = val_results.box.mp       # mean precision
        mr = val_results.box.mr       # mean recall
        map50 = val_results.box.map50
        map5095 = val_results.box.map

        metrics["precision_mean"] = round(float(mp), 4)
        metrics["recall_mean"] = round(float(mr), 4)
        metrics["mAP50"] = round(float(map50), 4)
        metrics["mAP50_95"] = round(float(map5095), 4)

        # Per-class metrics
        nc = len(TARGET_NAMES)
        header = f"\n{'Class':<16} {'Precision':>10} {'Recall':>8} {'mAP@50':>8} {'mAP@50-95':>10}"
        logger.info(header)
        logger.info("-" * 58)

        class_metrics = []
        for i in range(nc):
            try:
                p = float(val_results.box.p[i]) if hasattr(val_results.box, "p") else 0.0
                r = float(val_results.box.r[i]) if hasattr(val_results.box, "r") else 0.0
                ap50 = float(val_results.box.ap50[i]) if hasattr(val_results.box, "ap50") else 0.0
                ap = float(val_results.box.ap[i]) if hasattr(val_results.box, "ap") else 0.0
            except (IndexError, TypeError):
                p = r = ap50 = ap = 0.0

            class_metrics.append({
                "class": TARGET_NAMES[i],
                "precision": round(p, 4),
                "recall": round(r, 4),
                "mAP50": round(ap50, 4),
                "mAP50_95": round(ap, 4),
            })
            logger.info("%-16s %10.3f %8.3f %8.3f %10.3f", TARGET_NAMES[i], p, r, ap50, ap)

        logger.info("-" * 58)
        logger.info("%-16s %10.3f %8.3f %8.3f %10.3f", "MEAN", mp, mr, map50, map5095)

        metrics["per_class"] = class_metrics

        # Try to copy confusion matrix
        try:
            cm_candidates = list(val_results.save_dir.glob("confusion_matrix*.png"))
            if cm_candidates:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cm_dst = LOGS_DIR / f"confusion_matrix_{ts}.png"
                shutil.copy2(cm_candidates[0], cm_dst)
                logger.info("Confusion matrix saved -> %s", cm_dst)
                metrics["confusion_matrix"] = str(cm_dst)
        except Exception as exc2:  # noqa: BLE001
            logger.debug("Could not copy confusion matrix: %s", exc2)

    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse per-class metrics: %s", exc)

    return metrics


def save_eval_json(metrics: dict, logs_dir: Path, logger: logging.Logger) -> Path:
    """
    Save evaluation metrics to a timestamped JSON file.

    Args:
        metrics:  Metrics dict from run_evaluation().
        logs_dir: Directory to write the file.
        logger:   Logger instance.

    Returns:
        Path to the written JSON file.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = logs_dir / f"eval_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation results saved -> %s", out_path)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────────

def parse_args() -> tuple[argparse.Namespace, dict]:
    """
    Parse known CLI arguments and collect extra --key value pairs as config overrides.

    Returns:
        (known_args, overrides_dict) where overrides_dict maps config keys to
        their string values from the CLI.
    """
    parser = argparse.ArgumentParser(
        description="SmartSort YOLO11 training script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to train_config.yaml.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; evaluate models/best.pt on test set.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from models/last.pt.")

    known, unknowns = parser.parse_known_args()

    # Parse --key value pairs from unknowns
    overrides: dict = {}
    i = 0
    while i < len(unknowns):
        token = unknowns[i]
        if token.startswith("--"):
            key = token[2:]
            if i + 1 < len(unknowns) and not unknowns[i + 1].startswith("--"):
                overrides[key] = unknowns[i + 1]
                i += 2
            else:
                overrides[key] = "true"
                i += 1
        else:
            i += 1

    return known, overrides


# ── Main ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: orchestrate config loading, training, and evaluation."""
    args, overrides = parse_args()
    logger = setup_logging(LOGS_DIR)

    # ── Load + patch config ───────────────────────────────────────────────────────
    logger.info("Loading config: %s", args.config)
    cfg = load_config(args.config)

    if overrides:
        logger.info("CLI overrides: %s", overrides)
        cfg = apply_cli_overrides(cfg, overrides)

    if args.resume:
        resume_weights = MODELS_DIR / "last.pt"
        if resume_weights.exists():
            cfg["model"] = str(resume_weights)
            logger.info("Resume mode: starting from %s", resume_weights)
        else:
            logger.warning("--resume requested but models/last.pt not found. Starting fresh.")

    # Resolve relative paths in config against the project root (trashsort/)
    raw_data = cfg.get("data", "merged_dataset/dataset.yaml")
    data_yaml = str((PROJECT_ROOT / raw_data).resolve()) if not Path(raw_data).is_absolute() else raw_data
    cfg["data"] = data_yaml   # pass the resolved absolute path to YOLO

    raw_project = cfg.get("project", "smartsort/runs")
    if raw_project and not Path(raw_project).is_absolute():
        cfg["project"] = str((PROJECT_ROOT / raw_project).resolve())

    # ── Class weights ─────────────────────────────────────────────────────────────
    labels_train_dir = Path(data_yaml).parent / "labels" / "train"
    nc = 3
    class_weights = compute_class_weights(labels_train_dir, nc, logger)

    # ── Dataset stats ─────────────────────────────────────────────────────────────
    dataset_stats = count_dataset_stats(Path(data_yaml), logger)
    logger.info("Dataset stats: %s", dataset_stats)

    # ── W&B ───────────────────────────────────────────────────────────────────────
    wandb_run = init_wandb(cfg, class_weights, dataset_stats, logger)

    # ── Eval-only path ────────────────────────────────────────────────────────────
    if args.eval_only:
        model_path = MODELS_DIR / "best.pt"
        logger.info("--eval_only: skipping training.")
        metrics = run_evaluation(model_path, data_yaml, cfg, logger)
        save_eval_json(metrics, LOGS_DIR, logger)
        if wandb_run:
            log_metrics_to_wandb(wandb_run, {f"test/{k}": v for k, v in metrics.items()
                                              if not isinstance(v, (dict, list))}, logger)
        finish_wandb(wandb_run, logger)
        return

    # ── Training ──────────────────────────────────────────────────────────────────
    run_dir = run_training(cfg, class_weights, logger)

    # Copy best/last weights to smartsort/models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_pt = copy_weights(run_dir, MODELS_DIR, logger)

    # ── Post-training evaluation ──────────────────────────────────────────────────
    eval_model = best_pt or (MODELS_DIR / "best.pt")
    metrics = run_evaluation(eval_model, data_yaml, cfg, logger)

    if metrics:
        save_eval_json(metrics, LOGS_DIR, logger)
        if wandb_run:
            flat_metrics = {f"test/{k}": v for k, v in metrics.items()
                            if not isinstance(v, (dict, list))}
            log_metrics_to_wandb(wandb_run, flat_metrics, logger)

    finish_wandb(wandb_run, logger)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
