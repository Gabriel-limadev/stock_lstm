import joblib
from pathlib import Path

from src.config import *

from src.data.download import download_market_data
from src.data.preprocess import preprocess_data
from src.features.feature_engineering import create_features
from src.data.prepare_dataset import prepare_lstm_dataset
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.utils.plots import plot_predictions


def run_training_pipeline(stock: str):

    RAW_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    stock_name = stock.replace(".SA", "")
    model_directory = Path(MODELS_DIRECTORY) / stock_name
    model_directory.mkdir(
        parents=True,
        exist_ok=True
    )
    model_path = model_directory / MODEL_FILE
    scaler_path = model_directory / SCALER_FILE
    metadata_path = model_directory / METADATA_FILE

    print("=" * 60)
    print(f"Treinando modelo para {stock}")
    print("=" * 60)

    # =========================
    # Download
    # =========================

    print("1 - Baixando dados...")

    df = download_market_data(stock)

    df.to_csv(RAW_PATH, index=False)

    # =========================
    # Preprocessamento
    # =========================

    print("2 - Pré-processando...")

    df = preprocess_data(df)

    df.to_csv(PROCESSED_PATH, index=False)

    # =========================
    # Feature Engineering
    # =========================

    print("3 - Criando features...")

    df = create_features(df)

    df.to_csv(FEATURES_PATH, index=False)

    # =========================
    # Dataset LSTM
    # =========================

    print("4 - Preparando dataset LSTM...")

    X_train, y_train, X_test, y_test = prepare_lstm_dataset(
        df,
        scaler_path,
        FEATURES,
        TARGET,
        TRAIN_SIZE,
        WINDOW_SIZE
    )

    scaler = joblib.load(scaler_path)

    # =========================
    # Treinamento
    # =========================

    print("5 - Treinando modelo...")

    train_model(
        X_train,
        y_train,
        model_path,
        metadata_path,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        patience=PATIENCE,
        training_history_path=TRAINING_HISTORY_PATH
    )

    # =========================
    # Avaliação
    # =========================

    print("6 - Avaliando modelo...")

    y_pred, y_real, metrics = evaluate_model(
        model_path,
        scaler,
        X_test,
        y_test,
        FEATURES,
        TARGET
    )

    print("Resultados:")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    plot_predictions(
        y_real,
        y_pred,
        REPORT_PREDICTION_PATH
    )

    print("Pipeline finalizado com sucesso!")