"""
Configurações gerais do projeto.
"""

# ---------- stock ticker
STOCK = "PETR4.SA"

# ---------- PATHS
# Data
RAW_PATH = "data/raw/stock.csv"
PROCESSED_PATH = "data/processed/stock_processed.csv"
FEATURES_PATH = "data/processed/stock_features.csv"
# Model
SCALER_PATH = "src/models/scaler.pkl"
MODEL_PATH = "src/models/lstm_model.keras"
METADATA_PATH = "src/models/metadata.json"
# Reports
REPORTS_PATH = "src/reports"
TRAINING_HISTORY_PATH = "src/reports/training_history.csv"
REPORT_PREDICTION_PATH = "src/reports/prediction.png"

# ---------- DATASET
TRAIN_SIZE = 0.8
WINDOW_SIZE = 30

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

# ---------- LSTM
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT = 0.2
DENSE_UNITS = 16

# ---------- TREINAMENTO
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
PATIENCE = 10