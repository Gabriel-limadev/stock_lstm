from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)
import json
import pandas as pd
from src.models.lstm import build_lstm

def train_model(
    X_train,
    y_train,
    model_path,
    metadata_path,
    input_shape,
    epochs,
    batch_size,
    validation_split,
    patience,
    training_history_path

):
    """
    Treina o modelo LSTM.
    """

    model = build_lstm(input_shape)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    # Salva histórico do treinamento
    pd.DataFrame(history.history).to_csv(training_history_path, index=False)

    metadata = {
        "input_shape": list(input_shape),
        "epochs": epochs,
        "batch_size": batch_size,
        "features": X_train.shape[2]
    }

    # Salva o histórico de treinamento
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return history