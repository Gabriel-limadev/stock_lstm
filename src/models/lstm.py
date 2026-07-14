from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)
from src.config import LSTM_UNITS_1, LSTM_UNITS_2, DROPOUT, DENSE_UNITS


def build_lstm(input_shape):
    """
    Cria a arquitetura da LSTM.
    """

    model = Sequential()

    model.add(
        LSTM(
            units=LSTM_UNITS_1,
            return_sequences=True,
            input_shape=input_shape
        )
    )

    model.add(Dropout(DROPOUT))

    model.add(LSTM(units=LSTM_UNITS_2))

    model.add(Dropout(DROPOUT))

    model.add(Dense(units=DENSE_UNITS, activation="relu"))

    model.add(Dense(units=1))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model