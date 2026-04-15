"""
setup.py — One-time project setup for SmartSort.

Creates required directories, verifies Python dependencies, and prints
next-step instructions. Run this after cloning on a new machine:

    python setup.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SMARTSORT = ROOT / "smartsort"

DIRS_TO_CREATE = [
    ROOT / "roboflow_raw",
    ROOT / "merged_dataset" / "images" / "train",
    ROOT / "merged_dataset" / "images" / "val",
    ROOT / "merged_dataset" / "images" / "test",
    ROOT / "merged_dataset" / "labels" / "train",
    ROOT / "merged_dataset" / "labels" / "val",
    ROOT / "merged_dataset" / "labels" / "test",
    SMARTSORT / "models",
    SMARTSORT / "logs",
    SMARTSORT / "runs",
    SMARTSORT / "tests",
    SMARTSORT / "app" / "backend",
    SMARTSORT / "app" / "frontend",
]


def create_dirs() -> None:
    """Create all required directories that git cannot track (empty or gitignored)."""
    print("[1/3] Creating directories ...")
    for d in DIRS_TO_CREATE:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  {d.relative_to(ROOT)}/")
    print()


def install_deps() -> None:
    """Install Python dependencies from requirements.txt."""
    req = SMARTSORT / "requirements.txt"
    if not req.exists():
        print("[2/3] requirements.txt not found — skipping.")
        return

    print("[2/3] Installing dependencies ...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(req)],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print()
    except subprocess.CalledProcessError:
        print("  WARNING: pip install failed. Run manually:")
        print(f"    pip install -r {req}")
        print()


def write_dataset_yaml() -> None:
    """Write the default merged_dataset/dataset.yaml if it doesn't exist."""
    yaml_path = ROOT / "merged_dataset" / "dataset.yaml"
    if yaml_path.exists():
        print("[3/3] merged_dataset/dataset.yaml already exists — skipping.")
        return

    print("[3/3] Writing merged_dataset/dataset.yaml ...")
    content = (
        f"path: {str(ROOT / 'merged_dataset').replace(chr(92), '/')}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 3\n"
        "names:\n"
        "- Recyclable\n"
        "- Organic\n"
        "- General Waste\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"  Written: {yaml_path.relative_to(ROOT)}")
    print()


def print_next_steps() -> None:
    """Print instructions for the next steps."""
    print("=" * 60)
    print("Setup complete! Next steps:")
    print("=" * 60)
    print()
    print("  1. Download datasets:")
    print("     cd smartsort/scripts")
    print("     python download_roboflow.py --api_key YOUR_KEY")
    print()
    print("  2. Merge datasets:")
    print("     python merge_datasets.py --auto")
    print()
    print("  3. (Optional) Log in to Weights & Biases:")
    print("     wandb login YOUR_API_KEY")
    print()
    print("  4. Train:")
    print("     python train.py")
    print()
    print("  See SETUP.md for full documentation.")
    print()


def main() -> None:
    """Run the full setup pipeline."""
    print()
    print("SmartSort — Project Setup")
    print("=" * 60)
    print()
    create_dirs()
    install_deps()
    write_dataset_yaml()
    print_next_steps()


if __name__ == "__main__":
    main()
