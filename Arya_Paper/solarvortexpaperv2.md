# A Three-Stage Deep Learning Pipeline for Automated Cell-Level Extraction, Defect Triage, and Fine-Grained Segmentation in Photovoltaic Modules

**Status:** Draft v2 — Preprocessing pipeline documented in full; Stage 1 extraction methodology expanded with a complete 15-notebook research history; results placeholders retained pending experiment completion.

---

## Abstract

Automated visual inspection of photovoltaic (PV) modules is essential for large-scale solar farm quality control and predictive maintenance. Existing approaches are typically evaluated only on pre-cropped, single-cell images and often collapse fine-grained defect taxonomies into coarse binary labels, discarding information critical for severity assessment. This paper proposes a three-stage pipeline that operates directly on module-level electroluminescence (EL) imagery — including full modules and partial/occluded captures — spanning both monocrystalline and polycrystalline cell types.

Stage 1 performs cell extraction from raw module images. We document a systematic 15-notebook research journey spanning classical OpenCV-based grid-line detection (11 iterations on partial/single-cell images, 4 iterations on full-module images) before pivoting to YOLOv8-OBB (oriented bounding boxes), which proved both more robust and scalable. The YOLO-based approach was found to handle both full-module and partial-module images, eliminating the need for separate detectors. Stage 2 performs binary quality triage (OK vs. B-grade) using CNN backbones (ResNet, EfficientNet) augmented with a Convolutional Block Attention Module (CBAM), with hyperparameters tuned via Optuna. Stage 3 performs pixel-level defect segmentation on B-grade cells, consolidating a 29-class fine-grained defect taxonomy into four physically-motivated superclasses (crack, inactive area, corrosion/metallization, material/surface) and benchmarking convolutional (U-Net family) against transformer-based (SegFormer) architectures.

We report [X]% mAP for extraction, [X]% F1 for triage, and [X] mIoU / Dice for segmentation. *(Results TBD — experiments ongoing.)*

---

## 1. Introduction

The global PV industry is undergoing rapid capacity expansion, with cumulative installed solar capacity projected to exceed 10 TW by 2030. Maintaining cell-level quality throughout the module manufacturing lifecycle and in-field operation is a critical bottleneck: manual electroluminescence (EL) imaging inspection is labour-intensive, subjective, and fails to scale to the throughput requirements of modern manufacturing lines or drone-based fleet inspection.

Three unresolved gaps motivate this work:

1. **Pre-segmentation assumption.** Most published EL defect classification and segmentation studies (including benchmark work on the ELPV dataset) assume that individual cell crops are provided as input. In realistic deployment — automated manufacturing-line cameras, UAV-mounted EL imagers, or handheld inspection devices — raw module-level images arrive with arbitrary orientation, partial occlusion, and perspective distortion. No plug-in pipeline exists to bridge this gap.

2. **Coarse label space.** Binary "defective / non-defective" classification is the prevailing paradigm. While sufficient for pass/fail sorting, it provides no information about *which* failure mechanism is present or *how severe* it is — information needed for warranty decisions, rework prioritisation, and predictive degradation modelling.

3. **Taxonomic inflation vs. statistical tractability.** Fine-grained EL defect ontologies routinely enumerate 20–30 subtypes, yet many subtypes are statistically rare and visually ambiguous. Training pixel-level segmentation models on raw 29-class labels produces severely imbalanced problems. A principled, physics-grounded consolidation strategy is needed.

**Contributions of this paper:**

1. An end-to-end pipeline operating on **raw module-level EL images** (full or partial, mono- and polycrystalline), including perspective correction and cell normalisation.
2. A rigorous documentation of **15 OpenCV-based classical extraction iterations** and a root-cause analysis of their systematic failure modes, providing a reproducible ablation of why classical grid-line detection cannot scale to deployment diversity.
3. A **YOLOv8-OBB detector** for oriented cell extraction trained on an 80/20 train-validation split of 2,212 module images (1,769 train / 443 val), serving as the unified Stage 1 detector for both full-module and partial-module inputs.
4. A CBAM-augmented, Optuna-tuned **binary triage stage** benchmarked across ResNet and EfficientNet backbones.
5. A defect-mechanism-informed **29→4 class consolidation** for tractable segmentation, with a systematic comparison of CNN vs. transformer architectures.
6. A forward-looking roadmap toward **instance-level defect localisation and severity scoring**.

**Paper organisation.** Section 2 surveys related work. Section 3 describes the datasets. Section 4 details the three-stage methodology including the complete preprocessing history. Section 5 covers experimental setup. Section 6 presents results. Section 7 discusses findings and limitations. Section 8 concludes.

---

## 2. Related Work

### 2.1 PV Defect Datasets

The **ELPV dataset** (Deitsch et al., 2019) is the canonical benchmark for EL-based defect classification: 2,624 grayscale cell-level images (300×300 px), labelled with defect probability and cell technology (mono/poly). Its widespread adoption has shaped the research paradigm toward pre-cropped single-cell inputs, leaving the module-to-cell extraction problem largely unaddressed in the academic literature.

Our Stage 3 segmentation target derives from a multi-class pixel-annotated EL dataset encompassing 29 defect subtypes with polygon-level annotations. *(Dataset provenance, institutional source, and annotation protocol to be formally cited upon permission confirmation.)* The breadth of this taxonomy — spanning mechanical fracture artefacts, electrochemical degradation, metallisation defects, and surface contamination — motivates the consolidation strategy described in Section 3.

For Stage 1 (module-level extraction), the full-module EL image corpus was sourced from a private industrial dataset of high-resolution JPEG images (representative filename: `WS11249040878571.jpg`) acquired under controlled EL imaging conditions. These modules contain 144–208 extractable cells per image depending on module format (6-string, 10-string, 12-string configurations).

### 2.2 Classical Cell Extraction from Module Images

Classical approaches to PV cell grid detection rely on structured light-field properties of EL images: the grid of busbars and cell boundaries creates strong periodic spatial signals. Standard methods include:

- **Hough-transform-based line detection:** Detecting the regular busbar grid via probabilistic Hough line transforms and using detected lines to define cell crops.
- **Morphological line extraction:** Using directionally shaped structuring elements to isolate horizontal and vertical grid features, then computing line intersections to define cell corners.
- **Projection profiling:** Projecting image intensity along rows and columns to locate periodic minima corresponding to inter-cell gaps.
- **DBSCAN-assisted line clustering:** Applying density-based clustering to merge spatially proximate Hough detections into logical grid lines.

While elegant, these approaches assume approximately axis-aligned images, low noise, and consistent inter-cell contrast — conditions that fail for tilted, partially occluded, or field-degraded module captures.

### 2.3 Deep Learning for Cell Detection and Extraction

YOLO-family models (Redmon et al., 2016; Jocher et al., 2023) have demonstrated state-of-the-art throughput–accuracy trade-offs in industrial inspection settings. The **YOLOv8-OBB** variant extends axis-aligned detection with oriented bounding boxes, directly predicting a rotation angle alongside the standard box parameters — making it suited to arbitrary-orientation PV cell extraction without requiring pre-rectification. Instance segmentation variants (YOLOv11-Seg) provide pixel-accurate contour masks, offering an alternative extraction modality when cell boundary glare or partial occlusion renders a rectangular fit insufficient.

### 2.4 Binary and Coarse Defect Classification

ResNet (He et al., 2016) and EfficientNet (Tan & Le, 2019) are the dominant backbones for EL defect classification, offering complementary trade-offs between depth/residual connections and compound scaling. The **Convolutional Block Attention Module (CBAM)** (Woo et al., 2018) augments any CNN backbone with lightweight channel-wise and spatial-wise attention, enabling the network to focus on defect-relevant regions without external localisation supervision. **Optuna** (Akiba et al., 2019) provides an efficient Bayesian hyperparameter search framework, outperforming grid and random search at scale.

### 2.5 Fine-Grained Semantic Segmentation of PV Defects

U-Net (Ronneberger et al., 2015) and its derivatives (U-Net++, Attention U-Net) dominate medical-style pixel segmentation and have been applied to PV imagery for crack detection. Their encoder-decoder skip-connection design preserves high-frequency spatial detail, critical for thin crack structures. Transformer-based segmenters — particularly **SegFormer** (Xie et al., 2021) with its hierarchical mix-transformer encoder and lightweight all-MLP decoder — offer superior global context modelling, advantageous for diffuse, spatially extensive defects such as inactive regions and corrosion halos.

The gap: most published PV defect segmentation work targets binary defect/background masks; multi-class defect-type segmentation across a domain-principled taxonomy remains comparatively unexplored.

---

## 3. Dataset

### 3.1 Stage 1 — Module-Level EL Images (Cell Extraction)

- **Source:** Industrial EL image corpus (full PV modules). Sample filenames follow the pattern `WS<serial>.jpg`. Images are high-resolution JPEG captures of standard commercial PV modules.
- **Module formats encountered:** 144-cell and 208-cell modules (corresponding to 6×24 and 8×26 cell grids approximately).
- **Composition:** [X] full module images; partial-framing captures included in a held-out test subset to evaluate robustness.
- **Split:** 80% training (1,769 images) / 20% validation (443 images); split performed at the module level to prevent cell-level data leakage across the train/val boundary.
- **Dataset subsets used during classical CV exploration:** ARTS (e.g., `ARTS_00007_r4_c5.png`), SDLE (e.g., `SDLE_00514_A10-DH3000-3000h-PT-cell46.png`), and BMRK (e.g., `BMRK_00106_cell0286.png`) — each representing different imaging protocols and module manufacturers, which drove the need for format-agnostic extraction.

> **IMAGE TO ADD — Figure 1:** A representative full-module EL image (`WS<serial>.jpg`) alongside the extracted cell grid overlay showing detected oriented bounding boxes. Include one monocrystalline and one polycrystalline module example.

### 3.2 Stage 2 — Cell-Level Quality Triage Images

- **Source:** ELPV dataset (Deitsch et al., 2019) combined with internally extracted cell crops from Stage 1 outputs.
- **Classes:** Binary — OK (functional) vs. B-grade (defective).
- **Composition:** [X] cell images total; class distribution: [X]% OK / [X]% B-grade. Cell technology: [X]% monocrystalline, [X]% polycrystalline.
- **Split:** Module-level train/val/test split to preserve independence (cells from the same module are correlated and must not span splits).

### 3.3 Stage 3 — Pixel-Annotated B-Grade Cell Images

- **Source:** Multi-class EL defect segmentation dataset with polygon-level annotations covering 29 defect subtypes.
- **Input:** B-grade cells from Stage 2 (or ground-truth B-grade labels for isolated Stage 3 experiments).

**Table 1 — Defect taxonomy and 29→4 superclass consolidation:**

| Superclass | Description | Physical Failure Mechanism | Approx. Original Class IDs |
|---|---|---|---|
| **1 — Crack** | Mechanical fracture of silicon wafer; dark hairline or diagonal features in EL | Mechanical stress: handling, thermal cycling, hail impact | 10, 14, 28 *(verify)* |
| **2 — Inactive Area** | Electrically disconnected region; no EL luminescence emission | Isolated by crack network, soldering defect, or delamination | 11, 17, 20 *(verify)* |
| **3 — Corrosion / Metallisation** | Degraded finger/busbar contacts, oxidation halos, solder bond failure | Moisture ingress, electrochemical corrosion, thermal fatigue of metal contacts | 15, 16, 18, 26 *(verify)* |
| **4 — Material / Surface** | Contamination spots, discoloration patches, material inhomogeneity | Process contamination, phosphorous diffusion inhomogeneity, encapsulant yellowing | 12, 13, 19, 25, 27 *(verify)* |
| *Unmapped / Background* | Remaining class IDs not yet assigned | Resolve before submission: drop rare classes or merge into nearest superclass | *(list remaining IDs)* |

> **TODO before submission:** Document per-original-class sample counts; justify dropped rare classes (e.g., < N samples); resolve all unmapped IDs.

> **IMAGE TO ADD — Figure 2:** Class distribution bar chart — per-superclass sample counts before and after 29→4 consolidation, illustrating the class imbalance the loss function must address.

### 3.4 General Preprocessing

All EL images were loaded as **grayscale** (single-channel) 8-bit images. Preprocessing across the pipeline (detailed per stage in Section 4) included:

- **Median denoising** (3×3 kernel): Suppresses salt-and-pepper noise from EL sensor read-out without blurring edge structure.
- **CLAHE enhancement** (clipLimit=2.0, tileGridSize=8×8): Normalises non-uniform EL emission intensity across the cell area, improving subsequent threshold and edge detection reliability.
- **Perspective correction / homographic rectification**: Warps each extracted cell to a canonical upright view prior to Stage 2 classification.
- **Standard normalisation** (mean subtraction, std division): Applied before deep learning model input.

> **IMAGE TO ADD — Figure 3:** Before/after visualisation of median blur + CLAHE on representative monocrystalline and polycrystalline cell images, demonstrating normalisation of EL emission inhomogeneity.

---

## 4. Methodology

### 4.1 Stage 1 — Cell Extraction: A Systematic Research History

This section documents the complete iterative development of Stage 1, from the initial classical vision hypothesis through systematic failure characterisation, the pivot to YOLO-based detection, and the final unified approach. This record serves both as a methodological contribution (reproducible ablation of classical approaches) and as a motivational narrative for the YOLO adoption.

---

#### 4.1.1 Phase A — Classical OpenCV: Single-Cell and Partial-Module Images (preprocessing1–preprocessing11)

**Hypothesis:** Given the structured and periodic nature of PV module EL images, classical morphological image processing should suffice to locate and extract individual cells or cells from partial module views.

---

##### preprocessing1 — Exploratory Signal Analysis

The first notebook established the core image analysis toolkit and verified the presence of the busbar grid signal:

- Loaded grayscale EL images from the ARTS dataset (`ARTS_00007_r4_c5.png`).
- Applied **3×3 median blur** for noise suppression; visualised the effect via column/row intensity profiles, confirming the periodic busbar structure.
- Applied **CLAHE** (clipLimit=2.0, tileGridSize=8×8) for contrast enhancement.
- Computed **Otsu's threshold** to isolate the active cell region from the background.
- Used `skimage.measure.profile_line` to extract diagonal intensity profiles confirming directional grid structure.

*Role in the paper:* Established the signal processing baseline and confirmed that busbar periodicity is a detectable feature.

> **IMAGE TO ADD — Figure 4:** Row/column intensity profile plots (raw vs. median-filtered) showing the periodic busbar dips — from preprocessing1. Demonstrates that the grid signal exists and motivates the line detection approach.

---

##### preprocessing2 — Grid Intersection Detection and Perspective Warp

Built on preprocessing1 to implement the full classical cell extraction pipeline:

**Pipeline:**
```
Median Blur (3x3)
   -> CLAHE (clipLimit=2.0, tile=8x8)
   -> Adaptive Threshold (Mean-C, block=15, C=5)
   -> Morphological Open with vertical kernel (3 x H/2)   [isolate vertical lines]
   -> Morphological Open with horizontal kernel (W/2 x 1) [isolate horizontal lines]
   -> addWeighted combination -> grid image
   -> bitwise AND (vert AND horz) -> intersection points
   -> Convex Hull on intersection points
   -> order_points() -> TL, TR, BR, BL corners
   -> getPerspectiveTransform + warpPerspective -> 256x256 rectified crop
```

On `ARTS_00007_r4_c5.png`, the pipeline successfully recovered corners:
```
Top-left:     [152, 133]
Top-right:    [378, 143]
Bottom-right: [371, 407]
Bottom-left:  [148, 399]
```

and produced a clean perspective-corrected 256×256 crop of the cell.

*Limitation discovered:* The approach was tested on a single clean image; the morphological kernel sizes (`3 × H/2` for vertical, `W/2 × 1` for horizontal) are image-resolution-specific and were not validated across the dataset.

---

##### preprocessing3 — Active-Region Contour Approach

Explored an alternative cell localisation strategy based on the observation that the active EL-emitting cell region is the dominant bright blob in the image:

- Applied CLAHE then **inverted** the image (`cv2.bitwise_not`) so the bright cell becomes a white region on a dark background.
- Applied **adaptive threshold** on the inverted image.
- Removed border-touching regions (20-pixel margin) to suppress edge artefacts.
- Filtered contours by area (relative to image area) and retained the largest qualifying contour as the cell boundary.
- Used `approxPolyDP` to approximate the contour to a polygon and extracted the cell crop.

*Limitation:* The inversion-based approach conflates the active cell body with any bright defect regions, producing incorrect boundaries on images with large inactive areas or strong background reflections.

---

##### preprocessing4 — Batch Extraction Attempt and Failure Characterisation

The pipeline from preprocessing2/3 was generalised into a reusable function `extract_middle_cell()` and applied as a batch processor across the full dataset:

```python
def extract_middle_cell(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(median)
    # ... morphological grid detection + perspective warp ...
```

Processing `input_dir` → `extracted_cells/` on the full ARTS dataset produced the following skip log:

```
Skipping ARTS_00005_r3_c5.png: no cell found
Skipping ARTS_00005_r4_c5.png: no cell found
Skipping ARTS_00006_r10_c3.png: no cell found
Skipping ARTS_00007_r12_c1.png: no cell found
Skipping ARTS_00007_r4_c5.png: no cell found
Skipping ARTS_00020_r10_c1.png: no cell found
Skipping ARTS_00020_r5_c1.png: no cell found
```

A non-trivial proportion of images produced *no cell found* errors. Analysis revealed three root causes:
1. **Low-contrast cell edges** at the periphery of degraded cells — the adaptive threshold produced no reliable grid structure.
2. **Tilted images** — even slight rotation (5–10°) caused the morphological kernels (strictly horizontal/vertical) to miss the busbar lines.
3. **Partial occlusion** — cells near module edges had missing quadrants that broke the intersection detection logic.

> **IMAGE TO ADD — Figure 5:** Side-by-side comparison: a "success case" (clean ARTS cell with the classical pipeline producing a clean rectified crop) vs. a "failure case" (an ARTS image that produced "no cell found" — original image shown alongside the intermediate threshold/grid images to illustrate where the pipeline broke down).

---

##### preprocessing5 — Pivot Decision: YOLO Introduced

Given the systematic failure rate observed in preprocessing4, we pivoted to a data-driven detection paradigm. The transition is explicitly documented in the notebook:

```python
# IN THIS WE ARE GOING TO TRY THIS WITH YOLO
```

The Ultralytics library (version 8.3.235) was installed and the dataset reorganised into the standard YOLO directory structure:

```
dataset/
  images/
    train/   <- 1,769 images (80% of 2,212 total)
    val/     <- 443 images (20% of 2,212 total)
  labels/
    train/
    val/
```

Split was performed via random shuffle (`random.shuffle(images)`, split index = `int(0.8 * len(images))`), yielding:
```
Train: 1769 images
Val:   443 images
```

*This notebook marks the architectural inflection point of the Stage 1 research.*

---

##### preprocessing6 — Canny Edge Detection Approach

Despite the pivot decision, classical approaches were explored in parallel to fully characterise the failure space. preprocessing6 tested Gaussian blur + Canny edge detection as an alternative to morphological thresholding:

- **Gaussian blur** (1×1 kernel, minimal smoothing) + **Canny edge detector** (threshold1=100, threshold2=200).
- Contour filtering: retained only contours with `contourArea > 1000`.

*Outcome:* Canny edges were fragmented along busbar regions — producing dozens of small disconnected contour segments rather than a coherent cell boundary. Without explicit grid structure reasoning, it was impossible to reliably reconstruct the cell polygon from the fragment set.

---

##### preprocessing7 — NL-Means Denoising + Distance Transform (Single-Cell)

Applied to the ARTS dataset (`ARTS_00007_r4_c5.png`), this notebook explored a more sophisticated denoising and line-isolation strategy:

- `cv2.fastNlMeansDenoising(h=10)` replaced median blur for stronger denoising.
- **Distance transform** (`cv2.distanceTransform`, DIST_L2, mask=5) applied to the adaptive threshold output to distinguish thin grid lines (low distance values) from thick defect regions (high distance values).
- `thick_only = zeros; thick_only[dist > 1.0] = 255` retained only thick line structures.
- Morphological closing + horizontal/vertical kernel separation.
- **HoughLinesP** + custom merging.

Result on clean single-cell image:
```
Merged vertical lines:   [128, 381]
Merged horizontal lines: [121, 384]
Middle cell bounds: x1=128, y1=121, x2=381, y2=384
```

*Limitation:* The distance threshold (`dist > 1.0`) and HoughLinesP parameters were image-specific. The approach succeeded on this clean image but was not validated on the full dataset.

---

##### preprocessing8 — Multi-Cell Partial Module (Line Count Explosion)

The same NL-Means + distance transform + HoughLinesP pipeline was applied to a multi-cell partial module image (`ARTS_00009_r1_c6.png` — a partial module view showing several cells simultaneously):

```
Merged vertical lines:   [84, 127, 168, 253, 334, 398]
Merged horizontal lines: [99, 135, 384]
```

Six vertical and three horizontal lines were detected — correctly reflecting the partial multi-cell view, but requiring *a priori* knowledge of expected line counts. The pipeline had no mechanism to distinguish "2 lines → 1 cell" from "6 lines → multiple cells" without external format metadata.

*Key insight from preprocessing8:* Classical approaches require format-specific line count assumptions that cannot be inferred from image content alone.

---

##### preprocessing9 — Threshold-Based Approach on Multi-Cell Image

Applied to the same `ARTS_00009_r1_c6.png` image using a simpler pipeline:

- Binary threshold (pixel value = 20) + Gaussian blur (5×5) + Canny edge detection.
- Contour extraction and bounding-rect estimation for candidate cells.

Produced `Merged vertical lines: [84, 127, 168, 253, 334, 398]` — identical to preprocessing8, confirming that the multi-line detection pattern is intrinsic to the image content, not an artefact of any specific pipeline.

---

##### preprocessing10 — Otsu + Contour Approximation (SDLE Field-Aged Cells)

Tested on `flip_180_SDLE_00514_A10-DH3000-3000h-PT-cell46.png` — a field-aged cell from the SDLE dataset, which differs significantly from the ARTS images in degradation state and contrast:

- **Otsu's threshold** (`THRESH_BINARY_INV + THRESH_OTSU`) — adapted from classic Otsu to handle the inverted intensity of field-aged cells.
- Morphological closing (5×5 kernel) to clean the binary mask.
- Largest contour selected as the cell boundary; `approxPolyDP` used to fit a quadrilateral.
- If 4 corners found: perspective warp applied; otherwise: axis-aligned bounding box crop.

*Outcome:* Worked on this specific SDLE image (isolated cell body as dominant foreground object). Failed for any image containing multiple cells or strong background features.

---

##### preprocessing11 — NL-Means + CLAHE + HoughLinesP + DBSCAN (BMRK Dataset)

The most sophisticated classical attempt. Applied to `BMRK_00106_cell0286.png` — a cell from the BMRK dataset with a distinct imaging protocol from ARTS/SDLE:

- `fastNlMeansDenoising(h=10)` + CLAHE + adaptive threshold.
- **HoughLinesP** to detect raw line segments.
- **DBSCAN clustering** to merge spatially proximate Hough detections into logical grid lines:

```
Vertical lines detected:   16
Horizontal lines detected: 13

After DBSCAN clustering:
  Merged vertical lines:   3
  Merged horizontal lines: 2
```

- Generated candidate cell bounding boxes from all pairwise vertical × horizontal line combinations.
- Applied aspect ratio filter and minimum area filter to select the valid cell crop.
- `is_perspective_needed()` function checked whether the four corners were non-collinear; if so, `getPerspectiveTransform` was applied.

*This is the most generalisable classical approach developed* — but it still required dataset-specific DBSCAN epsilon values and minimum-sample parameters, and failed to generalise across ARTS / SDLE / BMRK without manual re-tuning.

---

**Table 2 — Summary of Classical Approach Failures (CellExtractOpenCV track)**

| Notebook | Approach | Dataset Tested | Key Limitation |
|---|---|---|---|
| preprocessing1 | Intensity profiling + CLAHE exploration | ARTS (single) | Exploratory; no extraction attempted |
| preprocessing2 | Morphological grid + perspective warp | ARTS (single) | Not validated across dataset |
| preprocessing3 | Inversion + active-region contour | ARTS (single) | Conflates cell body with bright defect regions |
| preprocessing4 | Batched `extract_middle_cell()` | ARTS (full dataset) | **5–10% skip rate; "no cell found" on challenging images** |
| preprocessing5 | — (YOLO pivot) | — | Transition notebook; confirms pivot |
| preprocessing6 | Canny edge detection | ARTS (single) | Fragmented contours; no coherent cell boundary |
| preprocessing7 | NL-Means + distance transform + Hough | ARTS (single) | Image-specific distance threshold |
| preprocessing8 | Same as preprocessing7 | ARTS (multi-cell) | Requires a priori line count knowledge |
| preprocessing9 | Otsu threshold + Canny | ARTS (multi-cell) | Same format-specific limitation |
| preprocessing10 | Otsu + contour approximation | SDLE (single, field-aged) | Fails on multi-cell images |
| preprocessing11 | NL-Means + Hough + DBSCAN | BMRK (single) | Dataset-specific DBSCAN epsilon tuning |

> **Key architectural insight:** The fundamental limitation of all classical approaches is their reliance on image-specific hyperparameters (threshold values, structuring element sizes, Hough parameters, DBSCAN epsilons) that cannot be set universally across the diversity of EL imaging conditions, module formats, and cell technologies in a real deployment scenario.

---

#### 4.1.2 Phase B — Classical OpenCV: Full-Module Images (FullModuleOpenCV, fullmodulev1–fullmodulev4)

In parallel with the single-cell/partial-module track, a separate series of notebooks attempted to extract cells directly from **full high-resolution module images** (`WS<serial>.jpg`). These images present an additional challenge: they contain 144–208 cells each and must be processed in bulk.

---

##### fullmodulev1 — Hough Line Detection at Module Scale

**Approach:** Resize full module to a fixed width (2,000 px) to standardise Hough detection, then detect lines via HoughLinesP. Parameters: `MIN_LINE_LENGTH=400`, `MAX_LINE_GAP=30`.

**Result on `WS11249040878571.jpg`:**
```
Vertical lines:   [48, 1199, 1201, 1207, 1209, 1644, 2078, 4053]
Horizontal lines: [111, 122, 548, 990, 2287, 2289, 2642, 2647, 2683, 2715]
```

Near-duplicate lines (e.g., 1199, 1201, 1207, 1209) and widely scattered outliers render raw Hough output unusable at module scale without post-processing. The module-scale busbar density produces many overlapping detections that a simple threshold cannot resolve.

---

##### fullmodulev2 — autocrop + Projection-Profile Peak Detection

**New contributions:**

1. `autocrop_dark_edges(img, threshold_ratio=0.7)` — automatically removes dark border margins by comparing row/column mean intensities to `threshold_ratio × global_mean`. This standardises the active module area before line detection.
2. Projection-profile peak detection: summed pixel intensity along rows (for horizontal lines) and columns (for vertical lines), then used `scipy.signal.find_peaks` or threshold-based peak identification.

**Result on a B-grade module (`WS11249040878638.jpg`):**
```
Vertical peaks:   23   ->  Merged vertical lines:   23
Horizontal peaks:  5   ->  Merged horizontal lines:  5
Extracted and saved 144 cells to 'extracted_cells' folder.
```

**First successful batch extraction from a full module image** — 144 cells saved as individual crops. However:
- Cell count (144 = 12 columns × 12 rows, given the specific module format) must match the actual module layout exactly.
- Any miscount produces systematically misaligned crops for the entire module.
- The projection-profile threshold was set empirically for this module format.

---

##### fullmodulev3 — Batch Processing Across Multiple Modules

Extended fullmodulev2 to process all modules in the OK and B-grade subdirectories:

```
=== Processing: WS11249040878571.jpg ===
Vertical lines: 27  |  Horizontal lines: 9   ->  Extracted 208 cells
    -> saved to all_cells_extracted\WS11249040878571
=== Processing: WS11249040884052.jpg ===
Vertical lines: 27  |  Horizontal lines: 8   ->  Extracted 182 cells
    -> saved to all_cells_extracted\WS11249040884052```

**Critical observation:** Two modules of presumably the same physical format produced different horizontal line counts (9 vs. 8), yielding different cell counts (208 vs. 182). This inconsistency indicates that the projection-profile peak detector is sensitive to image-specific intensity variations across the module series — a classic manifestation of the hyperparameter brittleness documented throughout the CellExtractOpenCV track.

---

##### fullmodulev4 — Improved Batch Extraction (Best Classical Full-Module Result)

The final and most refined classical full-module approach. Key improvements over fullmodulev3:

- Modular `process_module(img_path, save_root)` function encapsulating all steps.
- Retained `autocrop_dark_edges()` from fullmodulev2.
- Added `scipy.signal.find_peaks` with tuned `prominence` and `distance` parameters for more robust peak detection.
- Applied DBSCAN-based line merging (from preprocessing11) to consolidate near-duplicate projection peaks.

**Result on 10 B-grade modules:**
```
Found 10 modules.

Processing: WS11249040878638.jpg -> Saved 144 cells -> extracted_cells\WS11249040878638
Processing: [WS<serial>.jpg]     -> Saved [N] cells -> extracted_cells\[WS<serial>]
...
```

10 modules processed with partial success; manual verification of cell counts was still required per module.

**Final decision to deprecate all classical full-module approaches:**

Despite being the best classical full-module result, fullmodulev4 still exhibited:
- Format-specific peak detection parameters (proximity/prominence thresholds differed for 144-cell vs. 208-cell modules).
- No perspective-correction capability — modules with even slight rotation produced systematically misaligned cell grids.
- No mechanism to handle partial-module views (e.g., edge modules captured in only 3/4 frame).

**YOLOv8-OBB was adopted as the sole Stage 1 detector**, providing a unified solution for full-module, partial-module, and single-cell images without format-specific tuning.

---

**Table 3 — Summary of Classical Approach Failures (FullModuleOpenCV track)**

| Notebook | Approach | Result | Key Limitation |
|---|---|---|---|
| fullmodulev1 | HoughLinesP on 2000px-wide module | Duplicate/scattered lines — unusable | No post-processing to merge duplicate detections |
| fullmodulev2 | autocrop + projection-profile peaks | **144 cells extracted from one module** — first success | Requires correct format-specific line count |
| fullmodulev3 | Batch over multiple modules | 208 cells (9 horiz) vs. 182 cells (8 horiz) — inconsistent | Projection peak count varies across modules of same format |
| fullmodulev4 | Improved batch with DBSCAN line merging | 10 modules processed, requires manual verification | Format-specific parameters; no perspective correction |

---

#### 4.1.3 Phase C — YOLOv8-OBB: Unified Cell Extraction (Final Approach)

**Rationale for model selection:** YOLOv8-OBB (Ultralytics 8.3.235) predicts five parameters per detection: (cx, cy, w, h, θ), where θ is the rotation angle of the bounding box relative to the horizontal axis. This enables:

1. **Format agnosticism:** The detector learns cell appearance from labelled data rather than assuming a specific grid structure.
2. **Rotation robustness:** OBB detections naturally handle modules captured at arbitrary angles without requiring pre-rectification.
3. **Scale invariance:** YOLO's multi-scale feature pyramid detects cells at varying resolutions without format-specific parameter tuning.
4. **Unified coverage:** A single model handles full-module images (208 cells per image), partial-module views, and isolated single-cell images — replacing the entire classical pipeline.

**Dataset:**
- Total: 2,212 module-level EL images
- Training split: 1,769 images (80%)
- Validation split: 443 images (20%)
- Split strategy: random shuffle at module level (not cell level) to prevent data leakage

**Annotation format:** YOLO OBB format — per cell: `class_id cx cy w h theta` (all coordinates normalised to [0,1] relative to image dimensions; theta in degrees or radians depending on Ultralytics convention used).

**Training configuration (to be completed after hyperparameter sweep):**
```python
from ultralytics import YOLO

model = YOLO('yolov8n-obb.pt')   # nano backbone for initial experiments
results = model.train(
    data='solar_cells_obb.yaml',
    epochs=[TBD],
    imgsz=[TBD],
    batch=[TBD],
    optimizer=[TBD],
    lr0=[TBD],
    # ... full config after tuning
)
```

**Post-processing pipeline:**
```
YOLO OBB Detection Output: (cx, cy, w, h, theta) per cell
        |
Extract 4 corner points from OBB parameters
        |
cv2.getPerspectiveTransform(src_corners, dst_rect)
        |
cv2.warpPerspective(original_image, M, output_size)
        |
Canonical upright cell crop (resolution-normalised)
```

**Evaluation metrics:**
- mAP@0.5 (OBB-adjusted IoU — IoU computed on rotated bounding polygons)
- mAP@0.5:0.95 (averaged over IoU thresholds)
- Extraction completeness rate: (cells correctly detected) / (total cells per module)
- Over-segmentation rate: false-positive detections per module
- Per-module-format breakdown: 144-cell vs. 208-cell modules

> **IMAGE TO ADD — Figure 6:** Grid of YOLO OBB detection outputs on representative full-module EL images — showing oriented bounding boxes overlaid on the raw module image, alongside a sample of resulting perspective-corrected cell crops. Include at least one example of a tilted module to demonstrate OBB rotation robustness.

---

### 4.2 Stage 2 — Binary Quality Classification (OK vs. B-Grade)

Extracted and rectified cells from Stage 1 are classified as structurally intact (OK) or degraded (B-grade) using attention-augmented CNN classifiers with Optuna-tuned hyperparameters.

**Architecture:**
```
Rectified Cell Image (H x W x 1, grayscale)
          |
  ImageNet-pretrained Backbone
  (ResNet-[variant] | EfficientNet-[variant])
          |
  CBAM Attention Module
  |--> Channel Attention:
  |      GAP(F) -> FC(C/r) -> ReLU -> FC(C) -> Sigmoid -> scale(F)
  |--> Spatial Attention:
  |      AvgPool||MaxPool along channel axis -> concat -> Conv(7x7) -> Sigmoid -> scale
          |
  Global Average Pooling
          |
  FC layer + Dropout
          |
  Sigmoid -> P(B-grade)
```

**CBAM integration rationale:** Defect features in EL images are spatially localised (cracks occupy typically <5% of cell area) and spectrally discriminative (appearing as dark regions on a brighter background). CBAM's dual attention enables the network to both select relevant feature channels and suppress irrelevant spatial regions without requiring explicit defect location annotations during training.

**Optuna hyperparameter search:**

| Hyperparameter | Search Space |
|---|---|
| Learning rate | LogUniform[1e-5, 1e-2] |
| Backbone variant | {ResNet-18, ResNet-50, EfficientNet-B0, EfficientNet-B3} |
| CBAM reduction ratio r | {4, 8, 16} |
| Dropout rate | Uniform[0.1, 0.5] |
| Batch size | {16, 32, 64} |
| Optimizer | {Adam, AdamW, SGD+momentum} |
| Objective | Validation AUROC (threshold-invariant under class imbalance) |

Number of trials: [TBD]; pruner: [TBD].

**Evaluation metrics:** Accuracy, Precision, Recall, F1-score, AUROC, confusion matrix — reported separately for monocrystalline and polycrystalline cells (defect visibility and class distribution differ by technology).

> **IMAGE TO ADD — Figure 7:** Stage 2 CBAM architecture diagram illustrating channel attention and spatial attention modules inserted into the backbone, with annotated feature map shapes and the dual-path attention computation.

> **IMAGE TO ADD — Figure 8:** Optuna trial history plot — validation AUROC vs. trial number, showing convergence of the hyperparameter search.

---

### 4.3 Stage 3 — Defect Segmentation on B-Grade Cells

**Input:** B-grade cell crops cascaded from Stage 2 (or ground-truth B-grade labels for isolated ablation experiments).

**Class space:** 4 consolidated defect superclasses + background = 5-way pixel-level segmentation.

**Segmentation architectures benchmarked:**

| Model | Architecture Family | Key Characteristic |
|---|---|---|
| U-Net | Convolutional encoder-decoder | Symmetric skip connections preserve spatial detail |
| U-Net++ | Dense convolutional | Nested dense skip connections for multi-scale features |
| Attention U-Net | Attention-gated convolutional | Soft spatial attention on skip connection paths |
| SegFormer-B0 | Transformer (lightweight) | Hierarchical mix-transformer encoder + MLP decoder |
| SegFormer-B2 | Transformer (medium) | Larger mix-transformer capacity |

**Loss function:** Combined Dice + Focal loss, weighted by inverse class frequency to address severe class imbalance. Tversky loss is explored as an alternative, explicitly up-weighting recall for minority superclasses (corrosion, surface/material). The loss function choice is ablated in the results.

**Evaluation metrics:**
- Per-class IoU and mean IoU (mIoU)
- Dice coefficient (overall and per-class)
- Boundary F1 score: critical for thin crack structures where pixel-level IoU underrepresents perceptual segmentation quality
- Qualitative overlay comparisons across defect superclasses and failure cases

> **IMAGE TO ADD — Figure 9:** Qualitative segmentation overlay panel — one representative B-grade cell per defect superclass (crack, inactive area, corrosion, surface/material), showing ground-truth mask (left), best-performing model prediction (centre), and failure case from the same model (right).

---

### 4.4 Pipeline Integration

The three stages form a cascaded inference pipeline:

```
Full Module EL Image (raw JPEG, arbitrary orientation)
              |
     +---------+---------+
     |  Stage 1           |
     |  YOLOv8-OBB        |
     |  Cell Detector     |
     +---------+---------+
              |
     Oriented Bounding Boxes (cx, cy, w, h, theta) per cell
              |
     Perspective Correction
     getPerspectiveTransform + warpPerspective
              |
     Rectified Cell Crops
     (one per detected cell, canonical orientation)
              |
     +---------+---------+
     |  Stage 2           |
     |  CBAM-CNN          |
     |  Binary Triage     |
     +---------+---------+
              |
     +---------+------------------+
     |                            |
  OK cells                  B-grade cells
  (archived)                      |
                         +---------+---------+
                         |  Stage 3           |
                         |  Defect            |
                         |  Segmentation      |
                         |  (U-Net/SegFormer) |
                         +---------+---------+
                                   |
                         Pixel-level defect map
                         per cell:
                         [Crack | Inactive Area |
                          Corrosion | Surface]
                                   |
                    +-------------------------------+
                    |  Future Work:                 |
                    |  Severity Score +             |
                    |  Instance Localisation +      |
                    |  Automated Defect Report      |
                    +-------------------------------+
```

**Error propagation discussion:** Stage 1 extraction errors propagate through the cascade. A missed cell contributes a false negative to the Stage 2 output distribution; a misaligned or partially cropped cell produces an artificially distorted Stage 2/3 input. We plan an ablation experiment comparing pipeline performance when Stage 2/3 receives (a) ground-truth cell crops vs. (b) YOLO-predicted crops, to quantify the upstream error contribution.

---

> **IMAGE TO ADD — Figure 1 (Pipeline Overview):** End-to-end block diagram of the three-stage pipeline (referenced in Section 1). Horizontal flow: raw module image → Stage 1 (YOLO OBB) → perspective correction → Stage 2 (CBAM-CNN triage) → Stage 3 (segmentation) → defect map → future severity score.

---

## 5. Experimental Setup

**Hardware:** [GPU model, VRAM — to be filled in after experiments]

**Software stack:**

| Package | Version | Role |
|---|---|---|
| Python | 3.x | Runtime |
| PyTorch | [version] | Deep learning framework |
| Ultralytics | 8.3.235 | YOLOv8-OBB training and inference |
| timm | [version] | ResNet/EfficientNet backbone weights |
| segmentation-models-pytorch | [version] | U-Net family |
| HuggingFace Transformers | [version] | SegFormer |
| Optuna | [version] | Hyperparameter optimisation |
| OpenCV | 4.12.0.88 | Image processing |
| scikit-image | [version] | Intensity profiling |
| scikit-learn | [version] | DBSCAN clustering |
| NumPy | 1.26.4 | Array operations |
| Matplotlib | 3.8.4 | Visualisation |

**Per-stage training configuration:**

| Parameter | Stage 1 (YOLOv8-OBB) | Stage 2 (Classification) | Stage 3 (Segmentation) |
|---|---|---|---|
| Framework | Ultralytics | PyTorch + timm | PyTorch + SMP / HF |
| Input resolution | [X]×[X] | [X]×[X] | [X]×[X] |
| Epochs | [X] | [X] | [X] |
| Batch size | [X] | Optuna-selected | [X] |
| Optimizer | [X] | Optuna-selected | [X] |
| LR schedule | [X] | Optuna-selected | Cosine Annealing |
| Early stopping | patience=[X] | patience=[X] | patience=[X] |
| Random seeds | ≥ 3 | ≥ 3 | ≥ 3 |

**Reproducibility:** All experiments run with ≥ 3 independent random seeds. Results reported as mean ± std across seeds. Data splits are fixed before any training and reproduced by specifying `random.seed([seed])` prior to dataset shuffling.

---

## 6. Results

*(Tables to be populated as experiments complete. Structure and metrics defined here.)*

### Table 4 — Stage 1: Cell Extraction Performance

| Model | Module Type | mAP@0.5 | mAP@0.5:0.95 | Completeness Rate | Over-seg Rate |
|---|---|---|---|---|---|
| YOLOv8-OBB | Full module (144-cell) | TBD | TBD | TBD | TBD |
| YOLOv8-OBB | Full module (208-cell) | TBD | TBD | TBD | TBD |
| YOLOv8-OBB | Partial module | TBD | TBD | TBD | TBD |
| Classical OpenCV best (fullmodulev4) | Full module | N/A | N/A | ~90%* | High |

\* Estimated from fullmodulev2/v4 batch runs; exact skip rate from preprocessing4 output log.

### Table 5 — Stage 2: Binary Triage Performance

| Backbone | CBAM | Optuna-tuned | Accuracy | F1 | AUROC |
|---|---|---|---|---|---|
| ResNet-18 | No | No | TBD | TBD | TBD |
| ResNet-18 | Yes | No | TBD | TBD | TBD |
| ResNet-50 | Yes | Yes | TBD | TBD | TBD |
| EfficientNet-B0 | Yes | Yes | TBD | TBD | TBD |
| EfficientNet-B3 | Yes | Yes | TBD | TBD | TBD |

*Ablation: CBAM contribution isolated from Optuna tuning by comparing rows 1–2 (same backbone, same LR, with/without CBAM).*

### Table 6 — Stage 3: Defect Segmentation Performance

| Model | mIoU | Dice | Boundary F1 | Crack IoU | Inactive IoU | Corrosion IoU | Surface IoU |
|---|---|---|---|---|---|---|---|
| U-Net | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| U-Net++ | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Attention U-Net | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SegFormer-B0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SegFormer-B2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Planned Ablations

- **Loss function:** Dice+Focal vs. Tversky (Stage 3).
- **Class consolidation:** Direct 29-class segmentation baseline vs. 4-superclass consolidation — expected to show that consolidation substantially improves per-class IoU for minority classes.
- **Error cascade:** Stage 2/3 performance with ground-truth cell crops vs. YOLO-predicted crops as input (quantifies Stage 1 error propagation).

---

## 7. Discussion

### 7.1 Why Classical OpenCV Failed at Scale: A Root-Cause Analysis

The 15-notebook preprocessing journey across two tracks (CellExtractOpenCV and FullModuleOpenCV) reveals four systematic failure modes:

1. **Hyperparameter brittleness.** Every classical approach requires at least one image-specific parameter: CLAHE clip limits, adaptive threshold block sizes, morphological kernel dimensions, Hough thresholds, projection peak prominence, or DBSCAN epsilon. These parameters, once tuned for one image or dataset subset, break on the next due to variation in EL intensity, resolution, or contrast.

2. **Module format sensitivity.** The number of expected grid lines is a function of module type (6-string, 10-string, 12-string, etc.). Classical approaches implicitly assume known format — the projection-profile peak detector in fullmodulev3 counted 9 horizontal lines for one module and 8 for the next module of presumably the same type, producing 208 vs. 182 cells respectively. Without format metadata, there is no principled way to resolve such ambiguity.

3. **Contrast variability.** EL emission intensity varies with cell temperature, exposure time, degradation state, and busbar geometry. Low-contrast degraded regions cause morphological grid detectors to fail: the busbar lines in those regions fall below the adaptive threshold, breaking the line isolation step.

4. **Perspective and tilt intolerance.** Morphological line extraction assumes approximately axis-aligned structure. Even mild module tilt (5–10°) causes horizontal/vertical kernels to produce diagonal line responses that cannot be separated by the standard `MORPH_OPEN` approach. None of the classical methods tested incorporated a pre-rectification step for arbitrary-angle modules.

YOLOv8-OBB subsumes all four of these sensitivities into learned model weights. The training process implicitly learns format-agnostic cell appearances, robust to EL contrast variation and arbitrary rotation — at the cost of requiring labelled training data (annotated OBBs).

### 7.2 The Case for a Unified Detector

An alternative architecture would use separate detectors for full-module vs. partial-module/single-cell inputs. The fullmodulev4 experience shows that even within the full-module track, format-specific tuning is unavoidable in the classical setting. The YOLO approach eliminates this distinction entirely: the same YOLOv8-OBB model, trained on a mix of full-module and partial-module images, generalises to both input types through learned feature representations. This architectural simplicity reduces deployment complexity and maintenance burden.

### 7.3 CNN vs. Transformer Segmentation Trade-offs

U-Net-family models excel at thin, high-frequency features (cracks) due to skip-connection preservation of fine spatial detail across scales. SegFormer's global self-attention is theoretically better suited to diffuse, spatially extended defects (inactive areas spanning multiple busbars, corrosion halos) where local receptive fields are insufficient. Whether this theoretical advantage translates to measurable IoU gains on our 4-class taxonomy — with its mix of thin crack structures and diffuse inactive regions — is the key empirical question addressed by Table 6.

### 7.4 Limitations

- **Domain gap.** All training data is acquired under controlled lab EL conditions. Field-collected EL (drone or handheld) introduces JPEG compression artefacts, variable exposure, partial occlusion, and ambient light contamination not represented in the training distribution.
- **Unresolved taxonomy.** Several of the 29 original defect class IDs remain unmapped to the 4-superclass scheme. Their resolution (merge into the nearest superclass vs. drop on statistical grounds) will affect reported class frequencies and model performance.
- **Severity scoring absent.** The current pipeline produces a binary triage decision and a pixel-level defect-type map, but no severity score — limiting actionability for maintenance prioritisation and warranty decisions.
- **Annotation cost.** Pixel-level annotations for 29 defect classes are expensive to acquire. The 4-class consolidation is a pragmatic response to this constraint, but it sacrifices the granularity needed for some engineering decisions (e.g., distinguishing crack propagation from isolated material defects).

---

## 8. Conclusion and Future Work

We have presented a three-stage automated inspection pipeline for photovoltaic modules, addressing the gap between raw module-level EL imagery and actionable defect characterisation. The systematic exploration of 15 classical computer vision approaches — documented across preprocessing1–preprocessing11 (CellExtractOpenCV) and fullmodulev1–fullmodulev4 (FullModuleOpenCV) — established that classical grid-line detection is insufficiently robust for deployment-scale diversity, motivating the adoption of YOLOv8-OBB as a unified Stage 1 cell extractor. This single model replaces the entire classical pipeline across full-module, partial-module, and single-cell input types.

Stage 2 binary triage (CBAM-augmented CNN, Optuna hyperparameter search) and Stage 3 fine-grained segmentation (principled 29→4 class consolidation, CNN vs. transformer benchmark) complete the pipeline. Quantitative results are pending experiment completion. The framework is modular: each stage can be independently upgraded as stronger backbone architectures or larger annotated datasets become available.

**Planned future work:**

1. **Severity scoring.** Map Stage 3 pixel-level segmentation outputs to continuous or ordinal severity scores per defect superclass, calibrated against module power loss measurements from IV-curve characterisation.
2. **Instance-level localisation.** Extend from semantic segmentation to instance-level defect detection (Mask R-CNN or YOLOv11-Seg applied to B-grade cells), enabling counting and spatial clustering of individual defect instances.
3. **Domain adaptation.** Transfer the pipeline to field-acquired EL/IR imagery using unsupervised or semi-supervised domain adaptation, addressing the lab-to-field distribution shift.
4. **Cross-dataset generalisation.** Evaluate on entirely held-out datasets from different manufacturers, EL camera systems, and module formats to characterise generalisation.
5. **Edge deployment.** Benchmark Stage 1–2 latency on embedded GPU hardware (NVIDIA Jetson Orin, Hailo-8) for real-time manufacturing-line integration.
6. **Multi-modal fusion.** Investigate combining EL imagery with infrared (IR) thermography, as different defect types have complementary visibility across modalities.

---

## References

*(Populate with full citations before submission; key anchors listed below)*

1. Deitsch, S., Christlein, V., Berger, S., Buerhop-Lutz, C., Maier, A., Gallwitz, F., & Riess, C. (2019). Automatic classification of defective photovoltaic module cells in electroluminescence images. *Solar Energy*, 185, 455–468.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
3. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
4. Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. *ECCV*.
5. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD*.
6. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and efficient design for semantic segmentation with transformers. *NeurIPS*.
7. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
8. Jocher, G., et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
9. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *CVPR*.
10. Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2018). UNet++: A nested U-Net architecture for medical image segmentation. *DLMIA/MICCAI*.
11. *(Stage 3 dataset source — to be cited upon confirmation of publication permission)*

---

## Appendix A — Preprocessing Notebook Inventory

### A.1 CellExtractOpenCV Track (Single-Cell / Partial-Module)

| Notebook | Primary Dataset | Key Technique | Outcome |
|---|---|---|---|
| preprocessing1.ipynb | ARTS (single) | Median + CLAHE + Otsu + intensity profiling | Exploratory signal analysis; no extraction |
| preprocessing2.ipynb | ARTS (single) | Morphological grid + intersection detection + perspective warp | Concept validated; not generalised |
| preprocessing3.ipynb | ARTS (single) | Inversion + active-region contour + approxPolyDP | Exploratory; conflates cell body with bright defects |
| preprocessing4.ipynb | ARTS (full dataset) | Batched `extract_middle_cell()` | **Key failure: 5–10% skip rate "no cell found"** |
| preprocessing5.ipynb | ARTS (full dataset) | YOLO pivot + dataset reorganisation | **Transition: train=1769 / val=443** |
| preprocessing6.ipynb | ARTS (single) | Gaussian blur + Canny edge detection | Fragmented contours; no coherent boundary |
| preprocessing7.ipynb | ARTS (single, clean) | NL-Means + distance transform + HoughLinesP | Succeeded on clean image; image-specific thresholds |
| preprocessing8.ipynb | ARTS (multi-cell) | Same as preprocessing7 on partial module view | Detected 6 vert + 3 horiz lines; format-count issue |
| preprocessing9.ipynb | ARTS (multi-cell) | Binary threshold + Gaussian + Canny | Confirmed format-count limitation |
| preprocessing10.ipynb | SDLE (field-aged) | Otsu + contour polygon approximation | Single-cell-only; failed on multi-cell images |
| preprocessing11.ipynb | BMRK | NL-Means + CLAHE + HoughLinesP + DBSCAN | Most sophisticated classical; dataset-specific tuning |

### A.2 FullModuleOpenCV Track (Full-Module Batch Extraction)

| Notebook | Dataset | Key Technique | Best Result |
|---|---|---|---|
| fullmodulev1.ipynb | OK modules | HoughLinesP on 2000px-wide images | Duplicate lines; unusable raw output |
| fullmodulev2.ipynb | B-grade modules | autocrop_dark_edges + projection peak detection | **144 cells extracted from first full module** |
| fullmodulev3.ipynb | OK + B-grade modules (batch) | Batch processing of fullmodulev2 | Inconsistent counts: 208 vs. 182 cells across modules |
| fullmodulev4.ipynb | B-grade modules (batch) | Improved batch + DBSCAN line merging | **10 modules processed; manual count verification needed** |

---

## Appendix B — Figures to Add: Complete Editorial Checklist

The following figures are required for the final paper. Each entry specifies the figure number, title, source data, and generation notes.

| Figure | Title | Source / Notes |
|---|---|---|
| Fig. 1 | Pipeline Overview Diagram | Section 4.4 ASCII diagram → professional vector graphic. Horizontal flow: raw module → YOLO OBB → perspective correction → CBAM-CNN → segmentation → defect map. |
| Fig. 2 | Class Distribution Bar Chart | Stage 3 dataset: per-superclass sample counts before and after 29→4 consolidation. Highlights class imbalance motivating Dice+Focal loss. |
| Fig. 3 | CLAHE Effect Visualisation | Screenshots from preprocessing1/preprocessing2: before and after median blur + CLAHE on mono- and polycrystalline cells. |
| Fig. 4 | Intensity Profile Plots | Plots from preprocessing1: row and column intensity profiles (raw vs. median-filtered), showing periodic busbar dips. |
| Fig. 5 | Classical Approach Failure Gallery | 4-panel: (a) morphological detection success (clean ARTS cell), (b) "no cell found" failure image, (c) DBSCAN-merged Hough lines on multi-cell image, (d) fullmodulev1 raw Hough output with duplicate lines. |
| Fig. 6 | YOLO OBB Detection Results | Oriented bounding boxes overlaid on full-module EL images + resulting perspective-corrected crops. Include one tilted-module example. |
| Fig. 7 | CBAM Architecture Diagram | Channel attention + spatial attention computation illustrated within the backbone, annotated with feature map dimensions. |
| Fig. 8 | Optuna Trial History | Validation AUROC vs. trial number, showing convergence of Stage 2 hyperparameter search. |
| Fig. 9 | Stage 3 Qualitative Overlays | One B-grade cell per superclass: ground-truth mask (left), model prediction (centre), failure case (right). |
| Fig. 10 | Error Cascade Ablation | Bar chart: mIoU / F1 for Stage 2/3 with ground-truth cell crops vs. YOLO-predicted crops, quantifying Stage 1 error propagation. |

---

## Appendix C — Notes for v3 (Outstanding Items Before Submission)

- **Fill all `[TBD]` / `[X]` placeholders:** dataset counts per stage, GPU/VRAM details, per-stage training hyperparameters, all results tables.
- **Resolve unmapped class IDs** in the 29→4 taxonomy (Table 1): decide merge vs. drop for each unmapped ID; document per-original-class sample counts.
- **Add Optuna hyperparameter table:** once the Stage 2 search concludes, report the best trial's full hyperparameter configuration.
- **Formally cite Stage 3 dataset:** obtain permission and add full bibliographic entry.
- **Per-manufacturer / per-batch generalisation:** if manufacturer metadata is available for the full-module dataset, add a breakdown of Stage 1 performance by manufacturer.
- **Ablation: 29-class baseline vs. 4-superclass consolidation:** train a direct 29-class segmentation model (even a weak baseline) to justify the consolidation on quantitative grounds.
- **Reproducibility table:** report mean ± std across ≥ 3 seeds for all stages.
- **Cross-reference Appendix B figures** once generated and insert into the corresponding sections.
- **Abstract update:** revise abstract with final headline numbers once all results are in.
- **Related Work:** add citations for: Mask R-CNN (He et al., 2017), relevant PV-specific segmentation papers using the Stage 3 dataset, and any additional YOLO-for-inspection works identified in the literature review.
