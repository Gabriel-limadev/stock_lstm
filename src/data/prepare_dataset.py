import joblib
import pandas as pd
import numpy as np

from .dataset import (
    split_data,
    scale_data,
    create_sequences
)

def prepare_lstm_dataset(df: pd.DataFrame, scaler_path: str, features: list, target: str, train_size: float, window_size: int):

    train, test = split_data(df, train_size)

    train_scaled, test_scaled, scaler = scale_data(
        train,
        test,
        features
    )

    target_index = features.index(target)

    X_train, y_train = create_sequences( 
        train_scaled,
        window_size,
        target_index
    )

    X_test, y_test = create_sequences(
        test_scaled,
        window_size,
        target_index
    )

    joblib.dump(
        scaler,
        scaler_path
    )
   
    return (
        X_train,
        y_train,
        X_test,
        y_test
    )

def create_prediction_sequence(
    df: pd.DataFrame,
    scaler,
    features: list,
    window_size: int
):
    """
    Cria a última sequência utilizada pela LSTM para realizar uma previsão.
    """

    if len(df) < window_size:
        raise ValueError(
            f"É necessário possuir pelo menos {window_size} registros."
        )
    
    # Seleciona apenas as features utilizadas no treinamento
    data = df[features]

    # Escala os dados utilizando o scaler treinado
    data_scaled = scaler.transform(data)

    # Pega apenas a última janela
    last_sequence = data_scaled[-window_size:]

    # Adiciona a dimensão do batch
    X = np.expand_dims(last_sequence, axis=0)

    return X