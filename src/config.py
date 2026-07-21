"""
Configurações gerais do projeto.
"""
from pathlib import Path

# ==================================================
# PATHS
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = BASE_DIR / "data"

RAW_DIRECTORY = DATA_DIRECTORY / "raw"
PROCESSED_DIRECTORY = DATA_DIRECTORY / "processed"

RAW_PATH = RAW_DIRECTORY / "stock.csv"
PROCESSED_PATH = PROCESSED_DIRECTORY / "stock_processed.csv"
FEATURES_PATH = PROCESSED_DIRECTORY / "stock_features.csv"

MODELS_DIRECTORY = BASE_DIR / "models"

REPORTS_DIRECTORY = BASE_DIR / "reports"

TRAINING_HISTORY_PATH = REPORTS_DIRECTORY / "training_history.csv"
REPORT_PREDICTION_PATH = REPORTS_DIRECTORY / "prediction.png"

MODEL_FILE = "model.keras"
SCALER_FILE = "scaler.pkl"
METADATA_FILE = "metadata.json"

# ==================================================
# DATASET
# ==================================================
TRAIN_SIZE = 0.8
WINDOW_SIZE = 30

# ==================================================
# FEATURES
# ==================================================
# A ordem das features não deve ser alterada após o treinamento.
FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return",
    "SMA_10",
    "SMA_50",
    "RSI",
    "Volatility_10",
    "Volatility_50"
]

TARGET = "Close"

# ==================================================
# MODEL
# ==================================================
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT = 0.2
DENSE_UNITS = 16

# ==================================================
# TREINAMENTO
# ==================================================
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
PATIENCE = 10