# Species-Agnostic Plant Disease Classification (5-Class)

A transfer-learning pipeline for **general plant disease categories** that are visually distinct across many crop species. The model is designed to run **alongside a separate species classifier** (e.g., 250-class plant identification): species model → disease category → combined explanation in the UI (e.g., *Tomato* + *blight* → early/late blight symptoms).

This repository focuses on the **5-class disease model** (`healthy`, `powdery_mildew`, `rust`, `mold`, `blight`), trained from PlantVillage, PlantDoc, and iNaturalist sources, exported for **mobile deployment via TFLite**.

---

## Table of Contents

- [Motivation](#motivation)
- [Disease Taxonomy (5 Classes)](#disease-taxonomy-5-classes)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Training Strategy](#training-strategy)
- [Evaluation Results](#evaluation-results)
- [Known Confusions and Mitigations](#known-confusions-and-mitigations)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Training Workflow (Notebook)](#training-workflow-notebook)
- [Artifacts for Mobile](#artifacts-for-mobile)
- [Design Decisions and Iteration History](#design-decisions-and-iteration-history)
- [References and Datasets](#references-and-datasets)

---

## Motivation

PlantVillage and similar datasets label diseases **per crop** (e.g., *Tomato___Early_blight*, *Apple___Cedar_apple_rust*). That yields dozens of near-duplicate visual classes and poor generalization when paired with a species model in production.

This project **remaps** species-specific labels into a smaller set of **visually separable disease categories** that recur across plants:

| Category | Typical visual signal |
|----------|------------------------|
| `healthy` | Normal green tissue, no dominant lesion pattern |
| `powdery_mildew` | White powdery coating (külleme) |
| `rust` | Orange/brown pustules (pas) |
| `mold` | Velvety gray/green fungal growth on leaves |
| `blight` | Necrotic spots, halos, scorch-like lesions (yanıklık / leaf spots grouped) |

Classes such as `rot`, `scab`, `pest_damage`, and `chlorosis_yellowing` were **excluded** from the final 5-class setup after experiments showed high inter-class confusion or unreliable learning (especially `pest_damage`).

---

## Disease Taxonomy (5 Classes)

**Alphabetical label order** (used by Keras `image_dataset_from_directory` and saved JSON):

```json
["blight", "healthy", "mold", "powdery_mildew", "rust"]
```

| Index | Class | English | Notes |
|------:|-------|---------|--------|
| 0 | `blight` | Blight / leaf spot / bacterial spot (grouped) | Weakest class historically; targeted tuning |
| 1 | `healthy` | Healthy | Strong baseline accuracy |
| 2 | `mold` | Leaf mold / fungal mold | Tomato Leaf Mold, field mold images |
| 3 | `powdery_mildew` | Powdery mildew | Distinct white powder texture |
| 4 | `rust` | Rust | Distinct orange rust pustules |

---

## Architecture

| Component | Choice |
|-----------|--------|
| Backbone | **EfficientNetB3** (ImageNet weights) |
| Input size | **300×300×3** (matches EfficientNet-B3 default) |
| Head | GlobalAveragePooling2D → Dropout(0.4) → Dense(5, softmax) |
| Regularization | L2 (1e-4) on final Dense |
| Preprocessing | `tensorflow.keras.applications.efficientnet.preprocess_input` |

**Two-stage training**

1. **Stage 1 — Frozen backbone:** Train only the classification head (Adam lr=1e-3, up to 10 epochs, early stopping on `val_accuracy`).
2. **Stage 2 — Fine-tune:** Unfreeze top ~40% of backbone layers (Adam lr=1e-5, up to 25 epochs, early stopping).

---

## Data Pipeline

### Sources

| Source | Role | Notes |
|--------|------|--------|
| **PlantVillage** | Controlled lab images | Mapped via `pv_mapping`; capped per class to reduce lab dominance |
| **PlantDoc** | Field-style images | Mapped via `pd_mapping`; train/test split merged into val/test |
| **iNaturalist** | Extra field diversity | Research-grade observations; per-class search terms |

### Combined dataset

- Output folder: `data/disease_combined_5class_v1/`
- Splits: `train/`, `val/`, `test/` with one subfolder per disease class
- Manifest: `outputs/disease_image_manifest_5class.csv` (filename, source, split, disease)

### Balancing and caps

| Setting | Value | Purpose |
|---------|-------|---------|
| `MAX_TRAIN_PV_PER_CLASS` | 900 | Limit PlantVillage (`pv_*`) per class per split |
| `MAX_VAL_PV_PER_CLASS` | 400 | |
| `MAX_TEST_PV_PER_CLASS` | 450 | |
| `MAX_TRAIN_DOMINANT_CLASS` | 2800 | Cap `healthy` in train after all sources merged |
| `REBUILD_COMBINED_DIR` | `True` (first run) | Clean rebuild of combined folders |

### Label mapping refinements (blight)

To reduce **noisy blight labels**, these mappings were **removed** from training:

- PlantVillage: `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` (gray leaf spot ≠ classic blight)
- PlantDoc: `Corn Gray leaf spot`

Early/late blight, bacterial spot, septoria, and similar PV/PD classes remain mapped to `blight`.

---

## Training Strategy

### Augmentation (softened for lesion visibility)

Strong augmentation was reduced so **small blight lesions** are not washed out:

| Layer | Approx. setting |
|-------|-----------------|
| RandomFlip | horizontal only |
| RandomRotation | 0.18 |
| RandomZoom | 0.15 |
| RandomContrast | 0.18 |
| RandomBrightness | 0.14 |
| RandomTranslation | 0.08 |
| RandomSaturation / Hue | 0.15 / 0.05 |
| GaussianNoise | stddev 0.012 |

### Class weights

1. `sklearn.utils.class_weight.compute_class_weight("balanced", ...)`
2. **Blight-focused tuning** (final model):
   - `blight` weight × **1.35**
   - `healthy` weight × **0.92**

This reduces **blight → healthy** confusion at the cost of slightly more **other classes → blight** predictions (see confusion analysis below).

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| `BATCH_SIZE` | 32 |
| `SEED` | 42 |
| Stage 1 max epochs | 10 |
| Stage 2 max epochs | 25 |
| Optimizer (stage 1) | Adam 1e-3 |
| Optimizer (stage 2) | Adam 1e-5 |

---

## Evaluation Results

**Held-out test set** (combined pipeline, ~2,260 images):

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **94.29%** |
| **Weighted F1** | **0.9428** |
| **Macro F1** | 0.9422 |

### Per-class accuracy (test)

| Class | Accuracy | Correct / Total |
|-------|----------|-----------------|
| `healthy` | 97.8% | 485 / 496 |
| `powdery_mildew` | 96.7% | 446 / 461 |
| `rust` | 94.9% | 465 / 490 |
| `mold` | 91.0% | 253 / 278 |
| `blight` | 90.1% | 482 / 535 |

### Per-class precision / recall / F1 (sklearn report)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| `blight` | 0.923 | 0.901 | 0.912 |
| `healthy` | 0.913 | 0.978 | 0.944 |
| `mold` | 0.941 | 0.910 | 0.925 |
| `powdery_mildew` | 0.974 | 0.967 | 0.971 |
| `rust` | 0.969 | 0.949 | 0.959 |

Detailed exports: `outputs/per_class_accuracy_disease_5class.csv`, `outputs/classification_report_disease_5class.json`, `outputs/metrics_disease_5class.json`.

### Comparison to previous 5-class run (before blight tuning)

| Aspect | Previous | Current (tuned) |
|--------|----------|-----------------|
| Overall accuracy | ~93.5% | **94.29%** |
| `blight` accuracy | ~87.5% | **90.1%** |
| `blight` → `healthy` errors | 40 | **33** |
| Side effect | — | More `mold`/`rust` → `blight` confusions |

**Recommendation:** Use the **current tuned model** for deployment; keep a backup of the previous `.tflite` for A/B testing on device.

---

## Known Confusions and Mitigations

Typical high-error pairs on the test set:

| True | Predicted as | Approx. count |
|------|----------------|---------------|
| `blight` | `healthy` | 33 |
| `mold` | `blight` | 15 |
| `rust` | `blight` | 11 |
| `blight` | `mold` | 10 |
| `healthy` | `blight` | 9 |

**Mitigations in production (without retraining):**

- Apply a **confidence threshold**; show “uncertain” below e.g. 60–70%.
- Prefer **top-2 predictions** in the UI when margin is small.
- Combine with the **species model** for user-facing messages.

**If blight remains over-predicted:** soften weights (e.g. blight ×1.25, healthy ×0.95) and retrain, or use confidence gating for `blight` only.

---

## Project Structure

```
plant_project/
├── notebooks/
│   ├── plant_disease_10class_model.ipynb   # Main 5-class training & export pipeline
│   └── plant_disease_general_model.ipynb   # Earlier multi-class experiments
├── src/
│   └── paths.py                            # Central path definitions
├── data/
│   ├── raw/plantvillage/                   # PlantVillage (gitignored)
│   ├── raw/plantdoc/                       # PlantDoc (gitignored)
│   ├── inat_disease_images/                # iNaturalist downloads
│   └── disease_combined_5class_v1/         # train/val/test per class
├── models/
│   ├── disease_5class_best.keras           # Best checkpoint
│   ├── disease_5class_stage1.keras
│   ├── disease_5class.tflite               # Mobile export
│   └── disease_class_names_5class.json
├── class_names/
│   └── disease_class_names_5class.json
├── outputs/
│   ├── disease_image_manifest_5class.csv
│   ├── per_class_accuracy_disease_5class.csv
│   ├── classification_report_disease_5class.json
│   ├── metrics_disease_5class.json
│   └── training_history_disease_5class.json
├── requirements.txt
├── CLAUDE.md                               # Developer notes for agents
└── README.md
```

---

## Setup

### Requirements

- Python 3.10+ recommended (tested with 3.12)
- TensorFlow 2.x (notebook logs: 2.21.x)
- See `requirements.txt`

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Data (not in git)

1. **PlantVillage** — place under `data/raw/plantvillage/` (Kaggle: *PlantVillage dataset*).
2. **PlantDoc** — `data/raw/plantdoc/train` and `data/raw/plantdoc/test`.
3. **iNaturalist** — optional; notebook can download into `data/inat_disease_images/`.

---

## Training Workflow (Notebook)

Open and run **`notebooks/plant_disease_10class_model.ipynb`** in order:

| Step | Cell topic |
|------|------------|
| 1 | Imports, paths, `IMG_SIZE=(300,300)`, `COMBINED_DATASET_NAME` |
| 2 | 5 classes + `pv_mapping` / `pd_mapping` |
| 3–5 | iNaturalist constants, functions, download (optional) |
| 6 | Build `disease_combined_5class_v1` (set `REBUILD_COMBINED_DIR=True` first time) |
| 7 | `tf.data` + augmentation |
| 8 | Class weights (balanced + blight tuning) |
| 9 | EfficientNetB3 model |
| 10 | Stage 1 training |
| 11 | Stage 2 fine-tune |
| 12 | Evaluation + per-class metrics |
| 13 | Save `.keras` + class names JSON |
| 14 | **TFLite export** |
| 15 | TTA test (optional) |
| 16–18 | CSV/JSON reports, confusion matrix, field-proxy test (PlantDoc+iNat) |

**After changing mappings:** run cell 6 with `REBUILD_COMBINED_DIR = True`, then 7→13→14.

---

## Artifacts for Mobile

| File | Description |
|------|-------------|
| `models/disease_5class.tflite` | TensorFlow Lite model for on-device inference |
| `models/disease_class_names_5class.json` | Class names in **index order** (must match model output) |
| `class_names/disease_class_names_5class.json` | Same list (copy for app assets) |

**Inference checklist**

1. Resize input to **300×300** RGB.
2. Apply **EfficientNet `preprocess_input`** (same as training).
3. Read softmax output; map argmax index to JSON class list.
4. Optional: reject low-confidence predictions in the app layer.

**Backup:** Keep a copy of a previous `disease_5class.tflite` before re-exporting if you iterate on training.

---

## Design Decisions and Iteration History

| Version | Classes | Notes |
|---------|---------|--------|
| Early experiments | 10 | `bacterial`, `blight`, `healthy`, `leaf_spot`, `mold`, `pest_damage`, `powdery_mildew`, `rot`, `rust`, `viral` |
| v5.x notebook | 9 | Dropped `leaf_damage`; refined mappings |
| **Current production** | **5** | Visually distinct set; dropped rot/scab/pest/chlorosis; focus on deployable accuracy |

**Why 5 classes?** Fewer classes → less mutual confusion, higher per-class accuracy, simpler mobile UX when combined with a species model.

**Why not a two-stage healthy gate?** A binary *healthy vs diseased* model was considered but **removed** from the notebook to keep mobile integration to a **single TFLite file**.

---

## References and Datasets

- **PlantVillage** — [Kaggle: PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **PlantDoc** — field plant disease images (see paper/dataset source used in your thesis)
- **iNaturalist API** — research-grade observations for domain diversity
- **EfficientNet** — Tan & Le, EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

---

## License and Thesis Use

This codebase supports thesis work on **species-independent plant disease detection**. Cite dataset licenses (PlantVillage, PlantDoc, iNaturalist) according to their terms when publishing or deploying commercially.

For questions about reproducing results, start from `notebooks/plant_disease_10class_model.ipynb` and the saved files under `outputs/metrics_disease_5class.json`.
