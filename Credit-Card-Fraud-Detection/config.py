"""
config.py
Central configuration for all paths, constants, and reproducibility settings.
"""
from pathlib import Path

# Root
ROOT_DIR = Path(__file__).parent

# Data
DATA_RAW       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"

# Outputs
OUTPUTS_MODELS  = ROOT_DIR / "outputs" / "models"
OUTPUTS_FIGURES = ROOT_DIR / "outputs" / "figures"
OUTPUTS_REPORTS = ROOT_DIR / "outputs" / "reports"

# Logs
LOGS_DIR = ROOT_DIR / "logs"

# Reproducibility
RANDOM_STATE = 42

# Model
TARGET_COLUMN = "Class"
TEST_SIZE     = 0.2
CV_FOLDS      = 5
