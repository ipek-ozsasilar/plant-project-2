from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# PlantVillage dataset paths
PLANTVILLAGE_DIR = DATA_DIR / "plantvillage"
DISEASE_CLASS_NAMES_PATH = MODELS_DIR / "disease_class_names.json"

