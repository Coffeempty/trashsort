# SmartSort — Setup Guide

Real-time garbage classification into **Recyclable**, **Organic**, and **General Waste**
using YOLO11 detection on an NVIDIA RTX 3050 Ti (4 GB VRAM).

---

## Requirements

- Python 3.10+
- NVIDIA GPU with 4 GB+ VRAM (CPU works but training will be very slow)
- CUDA 11.8+ and matching PyTorch build
- Roboflow account + API key (free tier is enough)

---

## 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/trashsort.git
cd trashsort
```

---

## 2. Install dependencies

```bash
pip install -r smartsort/requirements.txt
```

> If you want a clean environment first:
> ```bash
> python -m venv .venv
> .venv\Scripts\activate      # Windows
> source .venv/bin/activate   # macOS / Linux
> pip install -r smartsort/requirements.txt
> ```

---

## 3. Set up Weights & Biases (optional but recommended)

Training metrics are logged to W&B automatically. Get a free API key at
https://wandb.ai/authorize, then run:

```bash
wandb login YOUR_API_KEY
```

If you skip this, training continues without W&B — no crashes.

---

## 4. Download datasets

Get your Roboflow API key from **roboflow.com → Account Settings → Roboflow API**.

```bash
cd smartsort/scripts
python download_roboflow.py --api_key YOUR_ROBOFLOW_KEY
```

Downloads three datasets into `roboflow_raw/`:

| Priority | Dataset | Images | Purpose |
|---|---|---|---|
| 1 | trash-detection-main | ~6 800 | Recyclable + General Waste coverage |
| 2 | food-waste | ~7 600 | Organic class coverage |
| 3 | trash-detections-fyp | varies | Extra recyclable variety |

To preview what will be downloaded without touching the network:

```bash
python download_roboflow.py --api_key YOUR_KEY --dry_run
```

---

## 5. Merge and balance datasets

Remaps all source class names to the 3-class target scheme, converts polygon
labels to bounding boxes, deduplicates, and oversamples underrepresented classes.

```bash
python merge_datasets.py --auto
```

The `--auto` flag skips the confirmation prompt. Drop it to review the class
remapping table before anything is written.

Output goes to `merged_dataset/` (ignored by git — regenerate locally).

---

## 6. Train

```bash
python train.py
```

Override any hyperparameter from `configs/train_config.yaml` on the command line:

```bash
python train.py --epochs 200 --batch 4
```

Resume a previous run:

```bash
python train.py --resume
```

Evaluate `models/best.pt` without retraining:

```bash
python train.py --eval_only
```

> **GPU note:** Default batch size is 8 with `yolo11m`. If you get a CUDA
> out-of-memory error the script automatically retries at batch 4. For GPUs
> with less than 4 GB VRAM run `python train.py --batch 2 --imgsz 512`.

---

## 7. Export

Export to ONNX and TorchScript, then benchmark latency:

```bash
python export_model.py
```

---

## 8. Evaluate

Full evaluation with confusion matrix, confidence histograms, annotated sample
predictions, and a Markdown report:

```bash
python evaluate.py
```

Results are saved to `smartsort/logs/`.

---

## Project structure

```
trashsort/
  smartsort/
    configs/
      classes.yaml          # class definitions + keyword mapping
      train_config.yaml     # all training hyperparameters
    scripts/
      download_roboflow.py  # dataset downloader
      merge_datasets.py     # merge + remap + balance
      train.py              # YOLO11 training
      export_model.py       # ONNX / TorchScript export + benchmark
      evaluate.py           # comprehensive evaluation + reports
    app/
      backend/              # FastAPI inference server (coming soon)
      frontend/             # Streamlit UI (coming soon)
    models/                 # trained weights (git-ignored)
    logs/                   # run logs and eval reports (git-ignored)
    runs/                   # YOLO training artifacts (git-ignored)
    requirements.txt
  roboflow_raw/             # downloaded datasets (git-ignored)
  merged_dataset/           # merged training data (git-ignored)
  TACO/                     # original TACO dataset (git-ignored)
```

---

## Classes

| ID | Class | Description |
|---|---|---|
| 0 | Recyclable | Glass, metal, paper, plastic, cardboard |
| 1 | Organic | Food waste, fruit, vegetables, meat, fish |
| 2 | General Waste | Mixed/non-recyclable waste, packaging |
