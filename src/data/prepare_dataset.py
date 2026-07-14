import joblib
import pandas as pd

from .dataset import (
    split_data,
    scale_data,
    create_sequences
)

def prepare_lstm_dataset(scaler_path: str, features_path: str, features: list, target: str, train_size: float, window_size: int):

    df = pd.read_csv(features_path)

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