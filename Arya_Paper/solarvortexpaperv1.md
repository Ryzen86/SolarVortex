# A Three-Stage Deep Learning Pipeline for Automated Cell-Level Extraction, Defect Triage, and Fine-Grained Segmentation in Photovoltaic Modules

**Status:** Rough Draft v1 — structural skeleton with placeholders for results, figures, and final numbers.

---

## Abstract

*(150–250 words — draft after Results are final)*

Automated visual inspection of photovoltaic (PV) modules is essential for large-scale solar farm quality control and predictive maintenance, but existing approaches are typically evaluated only on pre-cropped, single-cell images and often collapse fine-grained defect taxonomies into coarse binary labels, discarding information needed for severity assessment. This paper proposes a three-stage pipeline that operates directly on module-level electroluminescence (EL) imagery — including full modules and partial/occluded captures — under both monocrystalline and polycrystalline cell types. Stage 1 performs cell extraction from raw module images using YOLOv8-OBB (oriented bounding boxes) and YOLOv11-Seg, handling arbitrary module rotation and partial framing. Stage 2 performs binary quality triage (OK vs. B-grade) using CNN backbones (ResNet, EfficientNet) augmented with a Convolutional Block Attention Module (CBAM), with hyperparameters tuned via Optuna. Stage 3 performs pixel-level defect segmentation on B-grade cells, consolidating a 29-class fine-grained defect taxonomy into four physically-motivated superclasses (crack, inactive area, corrosion/metallization, material/surface) and benchmarking convolutional (U-Net family) against transformer-based (SegFormer) architectures. We report [X]% mAP for extraction, [X]% F1 for triage, and [X] mIoU / Dice for segmentation, and discuss a roadmap toward defect severity scoring and instance-level localization. *(Results TBD.)*

---

## 1. Introduction

- Motivation: scale of global PV deployment, cost of manual EL/IR inspection, throughput bottleneck in manufacturing QC and field O&M.
- Gap in existing literature:
  1. Most public work (e.g., ELPV dataset studies) assumes pre-segmented, axis-aligned, single-cell crops — not realistic for automated line inspection or drone/handheld field EL capture.
  2. Binary defective/non-defective classification is common, but doesn't tell an operator *what* is wrong or *how bad* it is.
  3. Fine-grained multi-class defect segmentation is rare because annotation is expensive and class imbalance is severe across 20+ defect subtypes.
- Contribution list (draft):
  1. An end-to-end pipeline that starts from **raw module-level images** (full or partial, mono- and polycrystalline) rather than pre-cropped cells.
  2. A **two-detector comparison** (YOLOv8-OBB vs. YOLOv11-Seg) for oriented cell extraction under module tilt/perspective.
  3. A CBAM-augmented, Optuna-tuned binary triage stage benchmarked across CNN backbones.
  4. A defect-mechanism-informed **29→4 class consolidation** for tractable segmentation, with a systematic comparison of CNN vs. transformer segmentation architectures.
  5. A discussion of the path from segmentation → **instance-level localization → severity scoring**, positioning this as a staged research program rather than a single model.
- Paper organization paragraph (standard).

---

## 2. Related Work

*(Each of these needs 1–2 paragraphs with real citations — placeholders below indicate what to cover)*

### 2.1 PV Defect Datasets
- ELPV dataset (Deitsch et al.) — 2,624 cell-level EL images, mono/poly, defect probability + type labels. Standard benchmark for binary/coarse classification.
- Larger multi-class EL/IR datasets with fine-grained pixel annotations (used for your 29-class taxonomy) — describe your source dataset explicitly here (name, size, annotation protocol, class list) since this is central to Stage 3.
- Note the field-vs-lab image domain gap: manufacturing-line EL vs. drone/handheld field EL/IR.

### 2.2 Detection/Extraction of Cells from Module Images
- Classical approaches: Hough transform / grid-line detection for cell segmentation from EL images.
- YOLO family for object detection in industrial inspection; OBB variants for rotated/tilted captures.
- Instance segmentation (Mask R-CNN, YOLO-Seg) for irregular/partial module framing.

### 2.3 Binary/Coarse Defect Classification
- CNN backbones (ResNet, VGG, EfficientNet) applied to ELPV and similar datasets.
- Attention mechanisms in defect classification: CBAM, SE-blocks — cite Woo et al. 2018 for CBAM.
- Hyperparameter optimization via Optuna (Akiba et al. 2019) vs. grid/random search — justify why Optuna is used.

### 2.4 Fine-Grained/Semantic Segmentation of PV Defects
- U-Net and variants (U-Net++, Attention U-Net) in industrial/medical-style pixel segmentation, and their applicability to sparse, thin, low-contrast defects (cracks).
- Transformer-based segmentation: SegFormer (Xie et al. 2021), Mask2Former — motivate why attention/global context matters for diffuse defects (e.g., inactive regions, corrosion) vs. thin local defects (cracks).
- Gap: most PV segmentation work stays at binary defect/no-defect masks; multi-class defect-type segmentation is comparatively unexplored — position your work here.

---

## 3. Dataset

- **Source(s):** name each dataset/imaging source used per stage (module-level images for Stage 1; cell crops with binary labels for Stage 2; pixel-annotated B-grade cells for Stage 3). If self-collected/field data supplements a public dataset, describe acquisition (EL vs. IR, camera, exposure settings).
- **Composition:** counts by module type (full/partial), cell technology (mono/poly), train/val/test split methodology (module-level split to avoid cell-level leakage — important to state explicitly, since cells from the same module are correlated).
- **Class taxonomy (Stage 3):** table listing all 29 original defect classes, sample counts per class, and your 4-class consolidation mapping with physical justification:

| Superclass | Original class IDs | Physical failure mechanism | Sample count |
|---|---|---|---|
| 1 — Crack | 10, 14, 28 | Mechanical fracture of silicon wafer | *(fill in)* |
| 2 — Inactive area | 11, 17, 20 | Electrically disconnected region (isolated by crack network or process defect) | *(fill in)* |
| 3 — Corrosion / Metallization | 15, 16, 18, 26 | Finger/busbar degradation, oxidation, solder bond failure | *(fill in)* |
| 4 — Material / Surface | 12, 13, 19, 25, 27 | Surface contamination, discoloration, material inhomogeneity | *(fill in)* |
| *(Unmapped)* | *(list remaining IDs)* | *(resolve: background / dropped-rare-class / merge target)* | *(fill in)* |

> **TODO before submission:** resolve and document the unmapped class IDs (see note above) and report per-original-class sample counts to justify consolidation and any dropped classes on statistical grounds (e.g., <N samples).

- **Preprocessing:** normalization, resizing/tiling strategy for high-resolution EL images, augmentation (rotation for OBB robustness, brightness/contrast jitter to simulate EL exposure variance, etc.).

---

## 4. Methodology

### 4.1 Stage 1 — Cell Extraction from Module Images
- Problem framing: input = full or partial module image (arbitrary orientation, mono/poly); output = individually cropped, orientation-normalized cell images.
- Model A: **YOLOv8-OBB** — rationale (handles rotated cells from perspective/tilted captures without over-cropping neighboring cells).
- Model B: **YOLOv11-Seg** — rationale (pixel-accurate cell boundary via instance segmentation, useful when busbar/cell-edge glare distorts a bounding-box fit).
- Post-processing: perspective correction / rotation normalization of extracted cells before Stage 2.
- Evaluation metrics: mAP@0.5, mAP@0.5:0.95 (OBB-adjusted IoU for the OBB model), mask IoU for the seg model, extraction completeness rate (cells missed per module) and over-segmentation rate.

### 4.2 Stage 2 — Binary Quality Classification (OK vs. B-Grade)
- Backbones: ResNet-[variant], EfficientNet-[variant].
- **CBAM integration:** where inserted (after which residual blocks / MBConv blocks), channel + spatial attention rationale for defect localization cues.
- **Optuna** search space: learning rate, backbone-specific hyperparameters, augmentation strength, optimizer choice; objective = validation F1 or AUROC (state which, and why, given likely class imbalance between OK/B-grade).
- Evaluation metrics: Accuracy, Precision/Recall, F1, AUROC, confusion matrix; report per-cell-technology (mono vs. poly) breakdown since defect visibility differs by technology.

### 4.3 Stage 3 — Defect Segmentation on B-Grade Cells
- Input: B-grade cells only (cascaded from Stage 2).
- Class space: 4 consolidated superclasses + background (5-way segmentation).
- Models compared: *(see Section 6 recommendations below — list final selection here once finalized)*.
- Loss function(s): justify choice given severe class imbalance (e.g., Dice + Focal combo, or Tversky loss weighted toward minority classes like corrosion).
- Evaluation metrics: per-class IoU, mean IoU, Dice coefficient, boundary F1 (important for thin crack structures where pixel-IoU underrepresents perceptual quality), and qualitative overlay comparisons.

### 4.4 Pipeline Integration
- End-to-end flow diagram: raw module image → Stage 1 extraction → Stage 2 triage → Stage 3 segmentation (B-grade only) → *(future: severity score + defect report)*.
- Error propagation discussion: how Stage 1 extraction errors affect Stage 2/3 (a placeholder for an ablation showing pipeline performance with ground-truth vs. predicted upstream boxes).

---

## 5. Experimental Setup

- Hardware/software stack (GPU, frameworks — PyTorch/Ultralytics for YOLO, timm/segmentation_models_pytorch or HuggingFace for SegFormer, Optuna version).
- Training schedule: epochs, batch size, optimizer, LR schedule, early stopping criteria, per stage.
- Reproducibility: seeds, number of runs per configuration (recommend ≥3 seeds, report mean ± std — this matters for PhD-level rigor and reviewer expectations).

---

## 6. Results

*(Tables/figures to fill in as experiments complete)*

- **Table 1:** Stage 1 extraction — YOLOv8-OBB vs. YOLOv11-Seg (mAP, mask IoU, completeness rate) by module type (full/partial) and cell technology.
- **Table 2:** Stage 2 triage — ResNet vs. EfficientNet, with/without CBAM, before/after Optuna tuning (Accuracy/F1/AUROC), plus an ablation isolating CBAM's contribution independent of tuning.
- **Table 3:** Stage 3 segmentation — model comparison (mIoU, Dice, boundary F1, per-class IoU) — see Section 8 for the recommended model set.
- **Figure:** qualitative segmentation overlays per defect superclass, including failure cases (e.g., thin cracks missed, corrosion boundary bleed).
- **Ablations:** loss function comparison; effect of the 29→4 consolidation vs. a hypothetical direct 29-class segmentation baseline (even a weak baseline here strengthens the paper's justification for consolidation).

---

## 7. Discussion

- Where each stage's errors concentrate and why (e.g., cracks under-segmented due to thin structure vs. thick-region defects segmented well).
- Trade-offs: CNN vs. transformer segmentation — inference cost vs. accuracy, relevant for deployment on inspection-line hardware.
- Limitations: dataset domain (lab EL vs. field conditions), unresolved/dropped classes, lack of severity/instance-level output (motivates Section 8).

---

## 8. Conclusion and Future Work

Summarize pipeline and headline numbers. State explicitly that segmentation is an intermediate milestone, with severity scoring and instance-level defect localization as the next research phase (detailed separately — see implementation roadmap).

---

## References

*(Populate with full citations before submission — key anchors to include)*

- Deitsch, S. et al. — ELPV dataset / automatic classification of defective PV cells using EL imaging.
- Woo, S. et al. (2018) — CBAM: Convolutional Block Attention Module.
- Akiba, T. et al. (2019) — Optuna: A Next-generation Hyperparameter Optimization Framework.
- Xie, E. et al. (2021) — SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.
- Ronneberger, O. et al. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation.
- Ultralytics YOLOv8/YOLOv11 documentation (for OBB and segmentation head architecture details).
- Relevant PV-specific fine-grained defect segmentation dataset paper (cite your Stage 3 data source explicitly).

---

## Appendix: Notes for v2

- Add exact hyperparameter tables once Optuna studies conclude.
- Add per-manufacturer / per-batch generalization test if metadata available.
- Resolve unmapped class IDs and add the completed class-frequency table.
- Insert final pipeline architecture diagram.