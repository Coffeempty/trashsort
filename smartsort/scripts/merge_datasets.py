"""
merge_datasets.py — Merge, remap, and balance multiple YOLO datasets into merged_dataset/.

Sources handled:
  - roboflow_raw/trash-detection-main/   (Roboflow, polygon/OBB labels → converted to bbox)
  - roboflow_raw/food-waste/             (Roboflow, standard bbox labels)
  - TACO/data/processed/                 (legacy 3-class, direct index remap)

Run from smartsort/scripts/:
    python merge_datasets.py --auto
    python merge_datasets.py --dry_run
    python merge_datasets.py --remap_override '{"food-waste:Other-waste": 2}'
"""

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Optional imports (show clear errors if missing) ─────────────────────────────
try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run: pip install pyyaml")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed. Run: pip install tqdm")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SMARTSORT_DIR = SCRIPT_DIR.parent
LOGS_DIR = SMARTSORT_DIR / "logs"
DEFAULT_TACO_DIR = SMARTSORT_DIR.parent / "TACO" / "data" / "processed"
DEFAULT_ROBOFLOW_DIR = SMARTSORT_DIR.parent / "roboflow_raw"
DEFAULT_OUTPUT_DIR = SMARTSORT_DIR.parent / "merged_dataset"
DEFAULT_CONFIG = SMARTSORT_DIR / "configs" / "classes.yaml"

# TACO old-class → new-class remap.
# old: 0=Dry(→Recyclable), 1=Wet(→Organic), 2=Hazardous(→skip)
TACO_REMAP: dict[int, Optional[int]] = {0: 0, 1: 1, 2: None}

TARGET_NAMES = ["Recyclable", "Organic", "General Waste"]
TARGET_NC = 3
SPLITS = ["train", "val", "test"]
BALANCE_TARGET_RATIO = 0.40  # oversample until each class ≥ 40% of the largest


# ─────────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────────

def setup_logging(logs_dir: Path) -> logging.Logger:
    """Configure logger to write to both stdout and a timestamped log file."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"merge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("merge_datasets")
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


# ─────────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────────

def load_classes_config(config_path: Path) -> tuple[dict[str, list[str]], list[str]]:
    """
    Load classes.yaml and return (subclass_keywords, skip_classes).

    Args:
        config_path: Path to classes.yaml.

    Returns:
        subclass_keywords: {target_class_name: [keyword, ...]}
        skip_classes: [class_name_to_skip, ...]
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["subclass_keywords"], [s.lower() for s in cfg.get("skip_classes", [])]


def load_roboflow_classes(dataset_dir: Path) -> list[str]:
    """
    Read class names from a Roboflow dataset's data.yaml.

    Args:
        dataset_dir: Root directory of the Roboflow dataset.

    Returns:
        List of class name strings.
    """
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("names", [])


# ─────────────────────────────────────────────────────────────────────────────────
# Label format detection and conversion
# ─────────────────────────────────────────────────────────────────────────────────

def polygon_to_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    """
    Convert a flat list of polygon/OBB coordinates [x1,y1,x2,y2,...] to YOLO bbox.

    Args:
        coords: Flat list of alternating x,y values (≥4 values, i.e. ≥2 points).

    Returns:
        (cx, cy, w, h) normalised to [0, 1].
    """
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def parse_label_line(line: str) -> Optional[tuple[int, float, float, float, float]]:
    """
    Parse one line of a YOLO label file into (class_id, cx, cy, w, h).

    Handles:
      - Standard bbox: class_id cx cy w h  (5 tokens)
      - OBB 4-corner:  class_id x1 y1 x2 y2 x3 y3 x4 y4  (9 tokens)
      - Polygon:       class_id x1 y1 x2 y2 ... (≥7 tokens, odd total)

    Returns None if the line is empty or malformed.
    """
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return None

    if len(values) == 4:
        # Standard YOLO bbox: cx cy w h
        cx, cy, w, h = values
    else:
        # Polygon or OBB: convert to axis-aligned bbox
        cx, cy, w, h = polygon_to_bbox(values)

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return class_id, cx, cy, w, h


# ─────────────────────────────────────────────────────────────────────────────────
# Keyword-based class remapping
# ─────────────────────────────────────────────────────────────────────────────────

def keyword_match(
    class_name: str,
    subclass_keywords: dict[str, list[str]],
) -> Optional[int]:
    """
    Match a source class name to a target class index using substring keyword search.

    On ambiguity (multiple categories match), the category whose longest matching
    keyword wins. If still tied, the first category in dict order wins.

    Args:
        class_name:        Source class name (will be lowercased internally).
        subclass_keywords: {target_class_name: [keyword, ...]}

    Returns:
        Target class index (0, 1, or 2) or None if no match.
    """
    name_lower = class_name.lower()
    best_target: Optional[str] = None
    best_kw_len = -1

    for target_name, keywords in subclass_keywords.items():
        for kw in keywords:
            if kw.lower() in name_lower:
                if len(kw) > best_kw_len:
                    best_kw_len = len(kw)
                    best_target = target_name

    if best_target is None:
        return None
    return TARGET_NAMES.index(best_target)


def build_remap_table(
    source_classes: list[str],
    subclass_keywords: dict[str, list[str]],
    skip_classes: list[str],
    overrides: Optional[dict[str, int]],
    dataset_name: str,
    logger: logging.Logger,
) -> dict[int, Optional[int]]:
    """
    Build a {source_index → target_index_or_None} mapping for one dataset.

    None means the class is skipped (not copied to merged dataset).

    Args:
        source_classes:    List of class names from the source dataset.
        subclass_keywords: Keyword mapping from classes.yaml.
        skip_classes:      List of names to always skip (lowercased).
        overrides:         Manual overrides from --remap_override CLI arg.
                           Keys are "dataset_name:class_name", values are target indices.
        dataset_name:      Used for override key lookup and logging.
        logger:            Logger instance.

    Returns:
        Mapping dict from source index to target index (or None to skip).
    """
    overrides = overrides or {}
    remap: dict[int, Optional[int]] = {}

    for src_idx, src_name in enumerate(source_classes):
        # Check manual override first
        override_key = f"{dataset_name}:{src_name}"
        if override_key in overrides:
            tgt = overrides[override_key]
            remap[src_idx] = tgt
            logger.debug("  Override: [%d] %s → %s", src_idx, src_name, TARGET_NAMES[tgt])
            continue

        # Check skip list
        if src_name.lower() in skip_classes:
            remap[src_idx] = None
            logger.debug("  Skip: [%d] %s (in skip_classes)", src_idx, src_name)
            continue

        # Keyword match
        tgt = keyword_match(src_name, subclass_keywords)
        if tgt is None:
            logger.warning(
                "  UNMAPPED: [%d] '%s' from '%s' — no keyword match, will be skipped.",
                src_idx, src_name, dataset_name,
            )
        remap[src_idx] = tgt

    return remap


def print_remap_table(
    remap: dict[int, Optional[int]],
    source_classes: list[str],
    dataset_name: str,
    logger: logging.Logger,
) -> None:
    """Print a formatted class-remapping table to the logger."""
    logger.info("")
    logger.info("  Remap table for '%s':", dataset_name)
    logger.info("  %-5s %-30s  →  %-5s %-20s", "SrcID", "Source Class", "TgtID", "Target Class")
    logger.info("  %s", "─" * 70)
    for src_idx, src_name in enumerate(source_classes):
        tgt = remap.get(src_idx)
        if tgt is None:
            tgt_str = "SKIP"
            tgt_name = "—"
        else:
            tgt_str = str(tgt)
            tgt_name = TARGET_NAMES[tgt]
        logger.info("  %-5d %-30s  →  %-5s %-20s", src_idx, src_name, tgt_str, tgt_name)


# ─────────────────────────────────────────────────────────────────────────────────
# Filesystem helpers
# ─────────────────────────────────────────────────────────────────────────────────

def find_split_dirs(dataset_dir: Path) -> dict[str, tuple[Path, Path]]:
    """
    Detect train/val/test split directories for a dataset.

    Handles two layouts:
      Roboflow: dataset_dir/{train|valid|test}/images/  + .../labels/
      TACO:     dataset_dir/images/{train|val|test}/    + .../labels/...

    Returns:
        {split_name: (images_dir, labels_dir)}  for each found split.
    """
    splits: dict[str, tuple[Path, Path]] = {}

    for split in SPLITS:
        # Roboflow uses "valid" for the val split
        folder_candidates = [split, "valid"] if split == "val" else [split]

        for folder in folder_candidates:
            # Roboflow style: {split}/images/
            img_dir = dataset_dir / folder / "images"
            lbl_dir = dataset_dir / folder / "labels"
            if img_dir.exists():
                splits[split] = (img_dir, lbl_dir)
                break

            # TACO style: images/{split}/
            img_dir2 = dataset_dir / "images" / folder
            lbl_dir2 = dataset_dir / "labels" / folder
            if img_dir2.exists():
                splits[split] = (img_dir2, lbl_dir2)
                break

    return splits


def hash_image(path: Path) -> str:
    """Return the MD5 hex-digest of a file's content."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk)
    return md5.hexdigest()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    """Return True if the file has an image extension."""
    return path.suffix.lower() in IMAGE_EXTS


# ─────────────────────────────────────────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────────────────────────────────────────

def process_label_file(
    label_path: Path,
    class_remap: dict[int, Optional[int]],
) -> Optional[list[tuple[int, float, float, float, float]]]:
    """
    Read a YOLO label file, remap class indices, convert polygons to bbox.

    Args:
        label_path:  Path to source .txt label file.
        class_remap: {src_class_id → target_class_id_or_None}

    Returns:
        List of (target_class_id, cx, cy, w, h) for kept boxes,
        or None if the label file could not be read.
        Returns an empty list if all boxes were skipped (caller should skip the image).
    """
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    kept: list[tuple[int, float, float, float, float]] = []
    for line in lines:
        parsed = parse_label_line(line)
        if parsed is None:
            continue
        src_cls, cx, cy, w, h = parsed
        tgt_cls = class_remap.get(src_cls)
        if tgt_cls is None:
            continue
        kept.append((tgt_cls, cx, cy, w, h))
    return kept


def write_label_file(
    path: Path,
    boxes: list[tuple[int, float, float, float, float]],
) -> None:
    """Write a list of (class_id, cx, cy, w, h) boxes to a YOLO label .txt file."""
    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, cx, cy, w, h in boxes]
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_split(
    img_dir: Path,
    lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    class_remap: dict[int, Optional[int]],
    prefix: str,
    split: str,
    dry_run: bool,
    logger: logging.Logger,
    stats: dict,
) -> None:
    """
    Copy one split of a source dataset into merged_dataset/, remapping labels.

    Args:
        img_dir:     Source images directory.
        lbl_dir:     Source labels directory.
        dst_img_dir: Destination images directory.
        dst_lbl_dir: Destination labels directory.
        class_remap: Class index remapping dict.
        prefix:      Short prefix added to filenames to avoid collisions.
        split:       Split name for logging ("train"/"val"/"test").
        dry_run:     If True, don't write anything.
        logger:      Logger instance.
        stats:       Mutable stats dict updated in place.
    """
    if not img_dir.exists():
        logger.debug("  %s: image dir not found, skipping: %s", split, img_dir)
        return

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in img_dir.iterdir() if is_image(p))
    if not images:
        logger.warning("  %s: no images found in %s", split, img_dir)
        return

    copied = 0
    skipped_no_labels = 0
    skipped_all_filtered = 0

    for img_path in tqdm(images, desc=f"  {prefix}/{split}", unit="img", leave=False):
        # Find matching label file (same stem, .txt extension)
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        if not lbl_path.exists():
            # Image without labels — skip (can't train without ground truth)
            skipped_no_labels += 1
            continue

        boxes = process_label_file(lbl_path, class_remap)
        if boxes is None:
            logger.warning("  Could not read label: %s", lbl_path)
            continue
        if len(boxes) == 0:
            # All boxes were filtered out — skip image to avoid blank samples
            skipped_all_filtered += 1
            continue

        new_stem = f"{prefix}_{img_path.stem}"
        dst_img = dst_img_dir / (new_stem + img_path.suffix)
        dst_lbl = dst_lbl_dir / (new_stem + ".txt")

        if not dry_run:
            try:
                shutil.copy2(img_path, dst_img)
            except OSError as e:
                logger.error("  Failed to copy image %s: %s", img_path, e)
                continue
            write_label_file(dst_lbl, boxes)

        # Update box-per-class stats for this split
        split_key = f"boxes_{split}"
        for cls, *_ in boxes:
            stats[split_key][cls] = stats[split_key].get(cls, 0) + 1

        stats[f"images_{split}"] = stats.get(f"images_{split}", 0) + 1
        stats[f"src_{prefix}_{split}"] = stats.get(f"src_{prefix}_{split}", 0) + 1
        copied += 1

    logger.info(
        "  %s/%s: copied %d, skipped %d (no labels), %d (all boxes filtered)",
        prefix, split, copied, skipped_no_labels, skipped_all_filtered,
    )
    if skipped_all_filtered > 0:
        stats["skipped_all_filtered"] = stats.get("skipped_all_filtered", 0) + skipped_all_filtered


def random_split_images(
    img_dir: Path,
    lbl_dir: Path,
    dst_base: Path,
    class_remap: dict[int, Optional[int]],
    prefix: str,
    dry_run: bool,
    logger: logging.Logger,
    stats: dict,
    seed: int = 42,
) -> None:
    """
    For flat datasets (no pre-defined splits), apply 80/10/10 random split then copy.

    Args:
        img_dir:  Flat images directory containing all images.
        lbl_dir:  Flat labels directory.
        dst_base: merged_dataset/ root.
        Others:   Same as copy_split().
    """
    images = sorted(p for p in img_dir.iterdir() if is_image(p))
    if not images:
        logger.warning("No images found in %s", img_dir)
        return

    rng = random.Random(seed)
    rng.shuffle(images)
    n = len(images)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    split_images = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split, split_imgs in split_images.items():
        dst_img_dir = dst_base / "images" / split
        dst_lbl_dir = dst_base / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for img_path in tqdm(split_imgs, desc=f"  {prefix}/{split}", unit="img", leave=False):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            boxes = process_label_file(lbl_path, class_remap)
            if not boxes:
                continue

            new_stem = f"{prefix}_{img_path.stem}"
            dst_img = dst_img_dir / (new_stem + img_path.suffix)
            dst_lbl = dst_lbl_dir / (new_stem + ".txt")

            if not dry_run:
                try:
                    shutil.copy2(img_path, dst_img)
                except OSError as e:
                    logger.error("Failed to copy %s: %s", img_path, e)
                    continue
                write_label_file(dst_lbl, boxes)

            for cls, *_ in boxes:
                split_key = f"boxes_{split}"
                stats[split_key][cls] = stats[split_key].get(cls, 0) + 1
            stats[f"images_{split}"] = stats.get(f"images_{split}", 0) + 1
            stats[f"src_{prefix}_{split}"] = stats.get(f"src_{prefix}_{split}", 0) + 1
            copied += 1

        logger.info("  %s/%s: copied %d images (random split)", prefix, split, copied)


# ─────────────────────────────────────────────────────────────────────────────────
# Post-merge operations
# ─────────────────────────────────────────────────────────────────────────────────

def deduplicate(output_dir: Path, dry_run: bool, logger: logging.Logger) -> int:
    """
    Remove duplicate images (identical file content) across all splits.

    Keeps the first occurrence found. Removes the duplicate image and its label.

    Returns:
        Number of duplicates removed.
    """
    logger.info("Running deduplication (MD5 hash check) …")
    seen: dict[str, Path] = {}
    removed = 0

    all_images: list[Path] = []
    for split in SPLITS:
        img_dir = output_dir / "images" / split
        if img_dir.exists():
            all_images.extend(sorted(p for p in img_dir.iterdir() if is_image(p)))

    for img_path in tqdm(all_images, desc="  Hashing", unit="img", leave=False):
        try:
            h = hash_image(img_path)
        except OSError as e:
            logger.warning("  Cannot hash %s: %s", img_path, e)
            continue

        if h in seen:
            logger.debug("  Duplicate: %s  (original: %s)", img_path, seen[h])
            lbl_path = img_path.parent.parent.parent / "labels" / img_path.parent.name / (img_path.stem + ".txt")
            if not dry_run:
                img_path.unlink(missing_ok=True)
                lbl_path.unlink(missing_ok=True)
            removed += 1
        else:
            seen[h] = img_path

    logger.info("  Deduplication: removed %d duplicate image(s).", removed)
    return removed


def count_boxes_per_class(labels_dir: Path, nc: int = TARGET_NC) -> dict[int, int]:
    """
    Count total bounding boxes per class in a labels directory.

    Args:
        labels_dir: Directory containing .txt label files.
        nc:         Number of target classes.

    Returns:
        {class_id: box_count}
    """
    counts: dict[int, int] = {i: 0 for i in range(nc)}
    if not labels_dir.exists():
        return counts
    for lbl in labels_dir.glob("*.txt"):
        try:
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parsed = parse_label_line(line)
                if parsed and parsed[0] in counts:
                    counts[parsed[0]] += 1
        except OSError:
            pass
    return counts


def flip_boxes(boxes: list[tuple[int, float, float, float, float]]) -> list[tuple[int, float, float, float, float]]:
    """
    Apply horizontal flip to a list of bounding boxes.

    Horizontal flip: cx → 1 - cx  (cy, w, h unchanged).
    """
    return [(cls, 1.0 - cx, cy, w, h) for cls, cx, cy, w, h in boxes]


def balance_classes(
    output_dir: Path,
    dry_run: bool,
    logger: logging.Logger,
    stats: dict,
) -> None:
    """
    Oversample underrepresented classes in the train split via horizontal flip augmentation.

    Target: each class should have at least BALANCE_TARGET_RATIO of the largest class's
    box count. Augmented copies are named <original>_aug<N>.{jpg,txt}.
    """
    train_img_dir = output_dir / "images" / "train"
    train_lbl_dir = output_dir / "labels" / "train"

    # Initial distribution
    counts = count_boxes_per_class(train_lbl_dir)
    if not any(counts.values()):
        logger.warning("Balance: no boxes found in train labels dir, skipping.")
        return

    logger.info("")
    logger.info("Class distribution BEFORE balancing (train split):")
    total = sum(counts.values()) or 1
    for cls_id, cnt in counts.items():
        logger.info("  [%d] %-15s  %6d boxes  (%5.1f%%)", cls_id, TARGET_NAMES[cls_id], cnt, cnt / total * 100)

    max_count = max(counts.values())
    target_count = int(max_count * BALANCE_TARGET_RATIO)

    # Build index: class_id → list of (img_path, lbl_path) that contain that class
    class_to_files: dict[int, list[tuple[Path, Path]]] = {i: [] for i in range(TARGET_NC)}
    for lbl_path in sorted(train_lbl_dir.glob("*.txt")):
        img_path = None
        for ext in IMAGE_EXTS:
            candidate = train_img_dir / (lbl_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
        try:
            classes_in_file = set()
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parsed = parse_label_line(line)
                if parsed:
                    classes_in_file.add(parsed[0])
        except OSError:
            continue
        for cls_id in classes_in_file:
            if cls_id in class_to_files:
                class_to_files[cls_id].append((img_path, lbl_path))

    aug_counts: dict[int, int] = {}

    for cls_id in range(TARGET_NC):
        current = counts[cls_id]
        if current >= target_count:
            continue  # Already balanced

        needed_boxes = target_count - current
        source_files = class_to_files.get(cls_id, [])
        if not source_files:
            logger.warning("  Class %d (%s): needs oversampling but no source images found.",
                           cls_id, TARGET_NAMES[cls_id])
            continue

        logger.info("  Oversampling class %d (%s): %d → target %d boxes",
                    cls_id, TARGET_NAMES[cls_id], current, target_count)

        added_boxes = 0
        aug_idx = 1
        rng = random.Random(cls_id * 1000)

        while added_boxes < needed_boxes:
            batch = source_files.copy()
            rng.shuffle(batch)
            for img_path, lbl_path in batch:
                if added_boxes >= needed_boxes:
                    break
                # Read original boxes
                try:
                    orig_lines = lbl_path.read_text(encoding="utf-8").splitlines()
                except OSError:
                    continue
                orig_boxes = [parse_label_line(l) for l in orig_lines if parse_label_line(l)]
                if not orig_boxes:
                    continue

                flipped = flip_boxes(orig_boxes)  # type: ignore[arg-type]
                new_stem = f"{lbl_path.stem}_aug{aug_idx}"
                new_lbl = train_lbl_dir / f"{new_stem}.txt"
                new_img = train_img_dir / f"{new_stem}{img_path.suffix}"

                if not dry_run:
                    write_label_file(new_lbl, flipped)
                    try:
                        shutil.copy2(img_path, new_img)
                    except OSError as e:
                        logger.error("  Augment copy failed: %s → %s: %s", img_path, new_img, e)
                        new_lbl.unlink(missing_ok=True)
                        continue

                # Count new boxes for this class
                new_cls_boxes = sum(1 for b in orig_boxes if b[0] == cls_id)
                added_boxes += new_cls_boxes
                aug_counts[cls_id] = aug_counts.get(cls_id, 0) + 1
                aug_idx += 1

    # Recount after balancing
    counts_after = count_boxes_per_class(train_lbl_dir)
    total_after = sum(counts_after.values()) or 1
    logger.info("")
    logger.info("Class distribution AFTER balancing (train split):")
    for cls_id, cnt in counts_after.items():
        added = aug_counts.get(cls_id, 0)
        aug_str = f"  (+{added} aug)" if added else ""
        logger.info("  [%d] %-15s  %6d boxes  (%5.1f%%)%s",
                    cls_id, TARGET_NAMES[cls_id], cnt, cnt / total_after * 100, aug_str)

    stats["aug_images"] = aug_counts


# ─────────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────────

def write_dataset_yaml(output_dir: Path, dry_run: bool, logger: logging.Logger) -> None:
    """Write merged_dataset/dataset.yaml for YOLO11 training."""
    yaml_path = output_dir / "dataset.yaml"
    content = {
        "path": str(output_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": TARGET_NC,
        "names": TARGET_NAMES,
    }
    logger.info("Writing dataset.yaml → %s", yaml_path)
    if not dry_run:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def print_summary(stats: dict, logger: logging.Logger) -> None:
    """Print the final merge summary."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("MERGE SUMMARY")
    logger.info("=" * 70)

    for split in SPLITS:
        n = stats.get(f"images_{split}", 0)
        logger.info("  %s images: %d", split, n)

    logger.info("")
    logger.info("  Bounding boxes per class per split:")
    logger.info("  %-16s %8s %8s %8s", "Class", "Train", "Val", "Test")
    logger.info("  %s", "─" * 50)
    for cls_id in range(TARGET_NC):
        tr = stats.get("boxes_train", {}).get(cls_id, 0)
        vl = stats.get("boxes_val", {}).get(cls_id, 0)
        te = stats.get("boxes_test", {}).get(cls_id, 0)
        logger.info("  [%d] %-13s %8d %8d %8d", cls_id, TARGET_NAMES[cls_id], tr, vl, te)

    logger.info("")
    logger.info("  Source contributions:")
    for key, val in stats.items():
        if key.startswith("src_") and val > 0:
            logger.info("    %s: %d images", key[4:], val)

    skipped = stats.get("skipped_all_filtered", 0)
    dups = stats.get("duplicates_removed", 0)
    logger.info("")
    logger.info("  Skipped (all boxes filtered): %d images", skipped)
    logger.info("  Duplicates removed:           %d images", dups)

    if stats.get("aug_images"):
        logger.info("")
        logger.info("  Augmented copies added per class:")
        for cls_id, n in stats["aug_images"].items():
            logger.info("    [%d] %s: +%d images", cls_id, TARGET_NAMES[cls_id], n)

    logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge + remap + balance Roboflow and TACO datasets into merged_dataset/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--taco_dir", type=Path, default=DEFAULT_TACO_DIR,
                        help="Path to TACO/data/processed/ (legacy 3-class format).")
    parser.add_argument("--roboflow_dir", type=Path, default=DEFAULT_ROBOFLOW_DIR,
                        help="Root directory containing downloaded Roboflow datasets.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Destination merged_dataset/ directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to configs/classes.yaml.")
    parser.add_argument("--auto", action="store_true",
                        help="Skip the y/n confirmation prompt after showing remap tables.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would happen without copying any files.")
    parser.add_argument("--remap_override", type=str, default=None,
                        help='JSON dict of manual class overrides, e.g. \'{"food-waste:Other-waste": 2}\'.')
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for flat-dataset splitting and oversampling.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: orchestrate the full merge pipeline."""
    # Ensure stdout can handle Unicode on Windows (cp1252 would choke on arrows/dashes)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    logger = setup_logging(LOGS_DIR)

    if args.dry_run:
        logger.info("=== DRY RUN - no files will be written ===")

    # ── Load config ───────────────────────────────────────────────────────────────
    logger.info("Loading classes config from %s", args.config)
    subclass_keywords, skip_classes = load_classes_config(args.config)

    overrides: dict[str, int] = {}
    if args.remap_override:
        try:
            overrides = json.loads(args.remap_override)
        except json.JSONDecodeError as e:
            logger.error("Invalid --remap_override JSON: %s", e)
            sys.exit(1)

    # ── Discover Roboflow datasets ────────────────────────────────────────────────
    roboflow_datasets: dict[str, Path] = {}
    if args.roboflow_dir.exists():
        for child in sorted(args.roboflow_dir.iterdir()):
            if child.is_dir() and (child / "data.yaml").exists():
                roboflow_datasets[child.name] = child
    logger.info("Roboflow datasets found: %s", list(roboflow_datasets.keys()))

    # ── Build all remap tables ────────────────────────────────────────────────────
    rf_remaps: dict[str, tuple[list[str], dict[int, Optional[int]]]] = {}
    for ds_name, ds_dir in roboflow_datasets.items():
        try:
            src_classes = load_roboflow_classes(ds_dir)
        except FileNotFoundError as e:
            logger.error("Cannot load classes for %s: %s", ds_name, e)
            continue
        remap = build_remap_table(src_classes, subclass_keywords, skip_classes, overrides, ds_name, logger)
        rf_remaps[ds_name] = (src_classes, remap)

    # ── Print remap tables ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("CLASS REMAPPING PLAN")
    logger.info("─" * 70)
    for ds_name, (src_classes, remap) in rf_remaps.items():
        print_remap_table(remap, src_classes, ds_name, logger)

    logger.info("")
    logger.info("  TACO (legacy 3-class direct remap):")
    logger.info("  %-5s %-20s  →  %-5s %-20s", "SrcID", "Source Class", "TgtID", "Target Class")
    logger.info("  %s", "─" * 60)
    taco_src_names = {0: "Dry (→Recyclable)", 1: "Wet (→Organic)", 2: "Hazardous (SKIP)"}
    for src_id, tgt_id in TACO_REMAP.items():
        tgt_str = str(tgt_id) if tgt_id is not None else "SKIP"
        tgt_name = TARGET_NAMES[tgt_id] if tgt_id is not None else "—"
        logger.info("  %-5d %-20s  →  %-5s %-20s", src_id, taco_src_names[src_id], tgt_str, tgt_name)

    # ── Confirmation ──────────────────────────────────────────────────────────────
    if not args.auto and not args.dry_run:
        logger.info("")
        answer = input("Proceed with merge? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            logger.info("Aborted by user.")
            sys.exit(0)

    # ── Prepare output directories ────────────────────────────────────────────────
    output_dir: Path = args.output_dir.resolve()
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats: dict = {
        "boxes_train": {},
        "boxes_val": {},
        "boxes_test": {},
    }

    # ── Process Roboflow datasets ─────────────────────────────────────────────────
    for ds_name, ds_dir in roboflow_datasets.items():
        if ds_name not in rf_remaps:
            continue
        src_classes, remap = rf_remaps[ds_name]
        # Shorten prefix to avoid very long filenames
        prefix = ds_name.replace("-", "")[:16]

        logger.info("")
        logger.info("Processing Roboflow dataset: %s  (prefix='%s')", ds_name, prefix)

        split_dirs = find_split_dirs(ds_dir)
        logger.info("  Splits detected: %s", list(split_dirs.keys()))

        if not split_dirs:
            # Flat dataset — look for images/ and labels/ directly
            flat_img = ds_dir / "images"
            flat_lbl = ds_dir / "labels"
            if flat_img.exists():
                logger.info("  No splits found — applying 80/10/10 random split.")
                random_split_images(flat_img, flat_lbl, output_dir, remap, prefix,
                                    args.dry_run, logger, stats, args.seed)
            else:
                logger.warning("  No images found in %s, skipping.", ds_dir)
            continue

        for split, (img_dir, lbl_dir) in split_dirs.items():
            dst_img_dir = output_dir / "images" / split
            dst_lbl_dir = output_dir / "labels" / split
            copy_split(img_dir, lbl_dir, dst_img_dir, dst_lbl_dir,
                       remap, prefix, split, args.dry_run, logger, stats)

    # ── Process TACO ──────────────────────────────────────────────────────────────
    taco_dir: Path = args.taco_dir.resolve()
    if taco_dir.exists():
        logger.info("")
        logger.info("Processing TACO dataset from %s", taco_dir)
        split_dirs = find_split_dirs(taco_dir)
        logger.info("  Splits detected: %s", list(split_dirs.keys()))

        for split, (img_dir, lbl_dir) in split_dirs.items():
            dst_img_dir = output_dir / "images" / split
            dst_lbl_dir = output_dir / "labels" / split
            copy_split(img_dir, lbl_dir, dst_img_dir, dst_lbl_dir,
                       TACO_REMAP, "taco", split, args.dry_run, logger, stats)
    else:
        logger.warning("TACO dir not found: %s — skipping.", taco_dir)

    # ── Deduplication ─────────────────────────────────────────────────────────────
    logger.info("")
    dups = deduplicate(output_dir, args.dry_run, logger)
    stats["duplicates_removed"] = dups

    # ── Class balancing ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("Running class balancing on train split …")
    balance_classes(output_dir, args.dry_run, logger, stats)

    # ── Write dataset.yaml ────────────────────────────────────────────────────────
    write_dataset_yaml(output_dir, args.dry_run, logger)

    # ── Final summary ─────────────────────────────────────────────────────────────
    print_summary(stats, logger)

    if args.dry_run:
        logger.info("DRY RUN complete — no files were written.")
    else:
        logger.info("Merge complete → %s", output_dir)


if __name__ == "__main__":
    main()
