
import joblib

from src.data.download import download_market_data
from src.data.preprocess import preprocess_data
from src.features.feature_engineering import create_features
from src.data.prepare_dataset import prepare_lstm_dataset
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.utils.plots import plot_predictions

from src.config import (
    STOCK, 
    RAW_PATH, PROCESSED_PATH, FEATURES_PATH, 
    FEATURES, TARGET, TRAIN_SIZE, WINDOW_SIZE, SCALER_PATH, 
    MODEL_PATH, METADATA_PATH, TRAINING_HISTORY_PATH, EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, PATIENCE,
    REPORT_PREDICTION_PATH
)

def main():

    # =========================
    # 1 - Download
    # =========================
    print("1 - Baixando dados...")

    download_market_data(STOCK, RAW_PATH)
    

    # =========================
    # 2 - Preprocessamento
    # =========================
    print("2 - Pré-processando...")

    preprocess_data(RAW_PATH, PROCESSED_PATH)

    # =========================
    # 3 - Feature Engineering
    # =========================
    print("3 - Criando features...")

    create_features(PROCESSED_PATH, FEATURES_PATH)

    # =========================
    # 4 - Preparação LSTM
    # =========================

    print("Preparando dataset LSTM...")

    X_train, y_train, X_test, y_test = prepare_lstm_dataset(
        SCALER_PATH,
        FEATURES_PATH,
        FEATURES,
        TARGET,
        TRAIN_SIZE,
        WINDOW_SIZE
    )

    scaler = joblib.load(
        SCALER_PATH
    )
    
    # =========================
    # 5 - Treinamento
    # =========================

    print("Treinando modelo...")

    train_model(
        X_train,
        y_train,
        MODEL_PATH,
        METADATA_PATH,
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
    # 6 - Avaliação
    # =========================

    print("Avaliando modelo...")

    y_pred, y_real, metrics = evaluate_model(
        MODEL_PATH,
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

if __name__ == "__main__":
    main()