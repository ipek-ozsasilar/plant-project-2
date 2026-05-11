# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Install dependencies into the virtual environment:
```bash
pip install -r requirements.txt
```

Download PlantVillage dataset from Kaggle:
```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset
```
Extract into `data/plantvillage/`.

Run the full training pipeline:
```bash
python src/train.py
```

Launch Jupyter for interactive work:
```bash
jupyter notebook notebooks/
```

## Architecture

This is a **general plant disease classification** system using transfer learning on PlantVillage dataset.

The goal is to detect **diseases** (not species) — e.g., bacterial spot, early blight, leaf scorch, healthy — that are common across many plant types. This approach is intentionally species-agnostic so the model can be used alongside a separate 250-class species model.

## Species-agnostic disease model strategy (project context)

We are building a **species-independent (türden bağımsız)** plant disease model. The key idea is: instead of learning “tomato early blight”, “apple scab”, etc. as separate classes, we learn **visual disease categories** that generalize across many plant species.

### Target label space (example)

We aim for ~8–12 general categories (final list may change based on data coverage):

- `healthy`
- `leaf_spot` (yaprak lekesi)
- `blight` (yanıklık)
- `powdery_mildew` (külleme)
- `rust` (pas)
- `mold` (leaf mold / küf)
- `rot` (çürüme)
- `chlorosis_yellowing` (sararma / kloroz)
- `pest_damage` (böcek zararı)

### Why this works

Many diseases share **similar visual patterns** across species (e.g., powdery mildew’s white powdery appearance). This enables a “disease category” model that can be paired with a separate **species model (250 classes)**.

### Two-model inference pipeline (app/product idea)

1) **Species model** → predicts plant species (250-class)
2) **Disease-category model** → predicts general disease category
3) Combine outputs in UI: “Domates” + “blight” → “Domateste yanıklık belirtileri var”

### Dataset strategy

We start with PlantVillage and optionally add “in-the-wild” datasets for robustness:

- **PlantVillage**: controlled images, many disease labels (good baseline)
- **PlantDoc**: field conditions, more realistic backgrounds/lighting
- **DiaMond / DIAMOND (tropical)**: useful for domain diversity (mango/banana etc.)
- **iNaturalist / observation-style sources**: optional, noisy but diverse (needs careful filtering)

### Relabeling / mapping approach

PlantVillage has many **species-specific** labels. We will remap them into the general categories above.

Example mapping snippet:

```python
disease_mapping = {
    "Apple___Apple_scab": "leaf_spot",
    "Apple___Black_rot": "rot",
    "Apple___Cedar_apple_rust": "rust",
    "Apple___healthy": "healthy",
    "Tomato___Early_blight": "blight",
    "Tomato___Late_blight": "blight",
    "Tomato___Leaf_Mold": "mold",
    "Tomato___Septoria_leaf_spot": "leaf_spot",
    "Tomato___healthy": "healthy",
}
```

This reduces dozens of classes into a smaller, more general label space.

### Notebook workflow requirement (important)

When implementing the pipeline in a notebook (`notebooks/*.ipynb`), keep cells **modular**:

- **Imports / config** in one cell
- **Dataset download/extract** in its own cell
- **Relabel/mapping utilities** in its own cell
- **Data loading** in its own cell
- **Model definition** in its own cell
- **Training** in its own cell (so running imports does not accidentally trigger training)
- **Evaluation + plots** separate
- **Export (TFLite / artifacts)** separate

Each cell should include a short markdown explanation of what/why it does.

### Model

EfficientNetV2S (ImageNet pretrained) backbone → GlobalAveragePooling2D → Dropout → Dense(N_classes, softmax). Input: 224×224×3.

### Data Pipeline

- `src/paths.py` — central `pathlib.Path` definitions for all project directories
- `src/train.py` — full end-to-end pipeline: data loading → augmentation → training → evaluation
- `data/plantvillage/` — PlantVillage dataset (gitignored)

### Saved Artifacts

All saved artifacts are gitignored:
- `models/best_disease_model.keras` — best model checkpoint
- `models/disease_class_names.json` — class names JSON array
- `outputs/history_disease_model.json` — training history
