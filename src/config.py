# src/config.py

from pathlib import Path

# Define the root directory of the project
ROOT_DIR = Path(__file__).parent.parent

# Define paths to other important directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
APP_DIR = ROOT_DIR / "app"

# Define paths to specific data files
RAW_DATASET_1 = RAW_DATA_DIR / "budgetwise_finance_dataset.csv"
RAW_DATASET_2 = RAW_DATA_DIR / "budgetwise_synthetic_dirty.csv"
PROCESSED_DATASET = PROCESSED_DATA_DIR / "cleaned_transactions.csv"

# Define paths for saved models
CHAMPION_MODEL_PATH = MODELS_DIR / "lightgbm_champion_model.pkl"