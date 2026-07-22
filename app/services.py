import joblib
from pathlib import Path
import pandas as pd
from pandas.tseries.offsets import BDay

from tensorflow.keras.models import load_model

from src.config import *

from src.data.download import download_market_data
from src.data.preprocess import preprocess_data
from src.features.feature_engineering import create_features
from src.data.prepare_dataset import create_prediction_sequence
from app.schemas import PredictionResponse

from src.pipeline.training_pipeline import run_training_pipeline


def train_model(stock: str):
    """
    Executa um novo treinamento do modelo.
    """

    try:
        run_training_pipeline(stock=stock)
    except Exception as e:
        raise ValueError(f"Erro ao treinar o modelo para {stock}: {str(e)}")

    return {
        "message": "Treinamento concluído com sucesso.",
        "stock": stock
    }

def predict_next_close(stock: str) -> PredictionResponse:
    """
    Realiza a previsão do próximo preço de fechamento para a ação informada.
    """

    stock = stock.strip().upper()
    
    if stock.endswith(".S"):
        stock += "A"
    elif not stock.endswith(".SA"):
        stock += ".SA"

    stock_name = stock.replace(".SA", "")
    model_directory = Path(MODELS_DIRECTORY) / stock_name
    model_path = model_directory / "model.keras"
    scaler_path = model_directory / "scaler.pkl"
    
    if not model_directory.exists():
        raise ValueError(
            f"Não existe um modelo treinado para {stock}."
        )
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    df = download_market_data(stock)
    if df.empty:
        raise ValueError(
            f"Não foi possível obter dados para {stock}."
        )
    df = preprocess_data(df)
    df = create_features(df)

    if len(df) < WINDOW_SIZE:
        raise ValueError(
            "Quantidade insuficiente de dados para realizar a previsão."
        )
    
    X = create_prediction_sequence(
        df=df,
        scaler=scaler,
        features=FEATURES,
        window_size=WINDOW_SIZE
    )

    prediction_scaled = model.predict(X)

    last_row = X[0, -1, :].copy()

    target_index = FEATURES.index(TARGET)

    last_row[target_index] = prediction_scaled[0][0]

    prediction = scaler.inverse_transform(
        last_row.reshape(1, -1)
    )[0][target_index]

    last_date = df["Date"].iloc[-1]
        
    prediction_date = last_date + BDay(1)

    return PredictionResponse(
        stock=stock,
        last_available_date=str(last_date.date()),
        prediction_date=str(prediction_date.date()),
        last_close=float(df["Close"].iloc[-1]),
        predicted_close=float(prediction)
    )
