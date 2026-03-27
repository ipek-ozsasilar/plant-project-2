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
