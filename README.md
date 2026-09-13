# SolarVortex

**Deep learning for photovoltaic electroluminescence (EL) inspection** — IIT Mandi research internship.

SolarVortex takes **raw module-level EL images** (full or partial, mono or poly, arbitrary orientation) and runs a staged pipeline: extract cells → triage OK vs B-grade → segment defect types on defective cells.

> For the full architecture, benchmarks, and file-level map, see [`PROJECT_DESCRIPTION.md`](PROJECT_DESCRIPTION.md).

---

## Pipeline

```
Raw module EL image (JPEG)
        │
        ▼
┌───────────────────────┐
│ Stage 1 · Extraction  │  YOLOv8-OBB (primary) · YOLOv8 · YOLO11-Seg
│                       │  Classical OpenCV notebooks (exploratory)
└───────────┬───────────┘
            │ upright cell crops
            ▼
┌───────────────────────┐
│ Stage 2 · Triage      │  OK vs B-grade · Optuna-tuned CNNs / ViTs ± CBAM
└───────────┬───────────┘
            │ B-grade only
            ▼
┌───────────────────────┐
│ Stage 3 · Segmentation│  CA-FUNet (prototype) · FFLUNet / nnU-Net (vendored)
└───────────────────────┘
```

Stages 1 and 2 are evaluated on **independent, non-overlapping** datasets to avoid leakage.

---

## Results (highlights)

### Stage 1 — Cell localization (YOLOv8-OBB)

| Model | mAP@50 | FPS (approx.) | Params |
|-------|--------|---------------|--------|
| **yolov8n-obb** | **0.995** | **~222** | ~3.1M |
| yolov8s-obb … x-obb | 0.995 | ~86–216 | ~11–70M |

### Stage 2 — Binary triage (OK / B-grade)

| Model | Test Acc | F1 Macro | AUC | Params |
|-------|----------|----------|-----|--------|
| **ConvNeXt-Tiny** | **97.2%** | **0.968** | 0.986 | 27.8M |
| DenseNet-121 | 96.9% | 0.964 | **0.989** | 7.0M |
| MobileNet-V3-Large | 96.0% | 0.954 | 0.970 | **3.0M** |

Full tables: `CellExtraction/YOLO/YOLOv8-OBB-aug/OBB_Benchmark/` and `BinaryDetection/Optuna/experimentA_split/benchmark_results/metrics/`.

---

## Repository layout

```
SolarVortex/
├── CellExtraction/          # Stage 1 — OpenCV + YOLO (AABB / OBB / seg)
├── BinaryDetection/         # Stage 2 — Optuna multi-model benchmark
├── CAFUNet/                 # Stage 3 — class-aware fusion U-Net prototype
├── FFLUNet/                 # Vendored nnU-Net v2 + FFLUNet trainers
├── Arya_Paper/              # Markdown drafts + IEEE iSPEC 2026 LaTeX (v1–v6)
├── frontend/                # Product vision README only (no app code)
└── PROJECT_DESCRIPTION.md   # Detailed project + architecture doc
```

| Component | Start here |
|-----------|------------|
| Stage 1 (OBB) | `CellExtraction/YOLO/YOLOv8-OBB-aug/cell_extract-yolov8-obb.ipynb` |
| Stage 2 | `BinaryDetection/Optuna/solar_cell_binary_benchmark.py` |
| Stage 3 | `CAFUNet/ca_funet.py`, `train_ca_funet.py` |
| Papers | `Arya_Paper/solarvortexpaperv2.md`, `Arya_Paper/IEEE-.../v6/iSPEC2026_SolarVortex.tex` |

---

## Tech stack

| Area | Tools |
|------|--------|
| Core | Python, PyTorch, torchvision, timm |
| Detection | Ultralytics YOLO (v8, v8-OBB, YOLO11-seg) |
| Classification | ResNet, EfficientNet, ConvNeXt, MobileNet, DenseNet, Swin, ViT · CBAM · Optuna |
| Classical CV | OpenCV, NumPy, scikit-learn |
| Segmentation | CA-FUNet · nnU-Net v2 / FFLUNet |
| Reporting | scikit-learn, matplotlib, seaborn, plotly |

---

## Quick start

There is no single root install — each stage is run independently.

### Stage 1 (YOLO)

```bash
# Create a Python env with torch + ultralytics, then open:
# CellExtraction/YOLO/YOLOv8-OBB-aug/cell_extract-yolov8-obb.ipynb
# Point data.yaml at your dataset and train with Ultralytics.
```

### Stage 2 (binary triage)

```bash
pip install torch torchvision optuna scikit-learn seaborn plotly kaleido pillow pandas
# Edit Config.DATA_ROOT in BinaryDetection/Optuna/solar_cell_binary_benchmark.py
# Expect DATA_ROOT/{train,val,test}/{Bgrade,ok}/
python BinaryDetection/Optuna/solar_cell_binary_benchmark.py
```

### Stage 3 (CA-FUNet smoke test)

```bash
pip install torch timm
python CAFUNet/test_ca_funet.py
```

### FFLUNet / nnU-Net (optional)

```bash
cd FFLUNet
pip install -e .
# Use nnUNetv2_* CLIs or scripts/*.sh (SLURM)
```

Large datasets and weights are gitignored / LFS (`*.pt`, `*.pth`). Place local data under paths referenced in notebooks or `Config`.

---

## Papers

- **IEEE iSPEC 2026 draft (two-stage):** *Two-Stage Deep Learning for PV Cell Localization and Defect Triage from Electroluminescence Images* — `Arya_Paper/IEEE-conference-template-062824/v6/`
- **Extended three-stage write-up:** `Arya_Paper/solarvortexpaperv2.md`

---

## Status

| Area | Status |
|------|--------|
| Stage 1 extraction + benchmarks | Done |
| Stage 2 Optuna triage + benchmarks | Done |
| Stage 3 CA-FUNet | Prototype |
| End-to-end inference service | Not implemented |
| Web product (`frontend/`) | Vision / README only |

---

## License / affiliation

Research work produced during an **IIT Mandi** internship. See individual subfolders (e.g. `FFLUNet/LICENSE`) for third-party package licenses.
