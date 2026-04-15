# SmartSort

Real-time garbage detection and classification using **YOLO11**. Detects objects in camera feeds or images and sorts them into three waste categories:

| ID | Class | Examples |
|---|---|---|
| 0 | **Recyclable** | Glass bottles, metal cans, paper, plastic, cardboard |
| 1 | **Organic** | Food scraps, fruit peels, vegetables, meat, fish bones |
| 2 | **General Waste** | Mixed waste, wrappers, non-recyclable packaging |

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/trashsort.git
cd trashsort
python setup.py
```

`setup.py` creates all required directories, installs pip dependencies, and writes the dataset config for your machine. Alternatively, do it manually:

```bash
pip install -r smartsort/requirements.txt
```

### 2. Download datasets

Get a free API key from [roboflow.com](https://roboflow.com) (Account Settings > Roboflow API).

```bash
cd smartsort/scripts
python download_roboflow.py --api_key YOUR_ROBOFLOW_KEY
```

This downloads ~14k images across two datasets into `roboflow_raw/`.

### 3. Merge and balance

```bash
python merge_datasets.py --auto
```

Automatically remaps 37 source classes to the 3-class target scheme, converts polygon annotations to bounding boxes, deduplicates across datasets, and oversamples underrepresented classes.

### 4. Train

```bash
python train.py
```

Default config: `yolo11m.pt`, batch 8, 150 epochs, 640px. Override anything on the CLI:

```bash
python train.py --epochs 200 --batch 4
python train.py --resume          # resume from last checkpoint
python train.py --eval_only       # evaluate models/best.pt without training
```

The script automatically retries at half batch size on CUDA OOM.

### 5. Export and evaluate

```bash
python export_model.py            # ONNX + TorchScript export with latency benchmark
python evaluate.py                # full eval: metrics, confusion matrix, sample predictions
```

## Pipeline Overview

```
download_roboflow.py     Roboflow SDK bulk download (YOLOv8 format)
        |
merge_datasets.py        Keyword-based class remap + polygon-to-bbox + dedup + balance
        |
train.py                 YOLO11 training with W&B logging + auto OOM fallback
        |
   +----+----+
   |         |
export_model.py    evaluate.py
ONNX/TorchScript   Per-class metrics, confusion matrix,
+ benchmark        confidence histograms, Markdown report
```

## Project Structure

```
trashsort/
  setup.py                          # one-command project setup
  smartsort/
    configs/
      classes.yaml                  # 3-class definitions + keyword mapping rules
      train_config.yaml             # all training hyperparameters
    scripts/
      download_roboflow.py          # dataset downloader (3 datasets, priority system)
      merge_datasets.py             # merge + remap + balance pipeline
      train.py                      # YOLO11 training + post-train eval
      export_model.py               # ONNX / TorchScript export + benchmark
      evaluate.py                   # comprehensive evaluation + report generation
    app/
      backend/                      # FastAPI inference server (planned)
      frontend/                     # Streamlit UI (planned)
    models/                         # trained weights (git-ignored)
    logs/                           # run logs, eval reports, plots (git-ignored)
    runs/                           # YOLO training artifacts (git-ignored)
    requirements.txt
  roboflow_raw/                     # downloaded datasets (git-ignored, ~15 GB)
  merged_dataset/                   # merged training data (git-ignored)
```

## Training Data

| Source | Images | Maps to |
|---|---|---|
| trash-detection-main (Roboflow) | ~6,800 | Glass/Metal/Paper/Plastic -> Recyclable, Waste -> General Waste |
| food-waste (Roboflow) | ~7,600 | 31 food classes -> Organic, Other-waste -> General Waste |
| TACO (local) | ~450 | Dry -> Recyclable, Wet -> Organic, Hazardous -> skipped |

After merge and balancing: **~14k train** / ~1.4k val / ~870 test images.

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 4 GB (batch 4-8) | 8 GB+ (batch 16+) |
| RAM | 8 GB | 16 GB+ |
| Disk | 30 GB (datasets + weights) | 50 GB |

Developed and tested on NVIDIA RTX 3050 Ti Laptop (4 GB VRAM), Windows 11, Python 3.13.

## Configuration

All training hyperparameters live in `smartsort/configs/train_config.yaml`. Key defaults:

| Parameter | Value | Notes |
|---|---|---|
| model | yolo11m.pt | YOLO11 medium |
| imgsz | 640 | Input resolution |
| batch | 8 | Max safe for 4 GB VRAM |
| epochs | 150 | With early stopping (patience=30) |
| cls | 1.5 | Raised from default 0.5 to penalise misclassifications |
| lr0 | 0.01 | Initial learning rate |

Class remapping rules live in `smartsort/configs/classes.yaml` — edit the keyword lists to change how source dataset classes are mapped to the three target categories.

## License

This project uses datasets under CC BY 4.0 (Roboflow) and their respective licenses (TACO).
