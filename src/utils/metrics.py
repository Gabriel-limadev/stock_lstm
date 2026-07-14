import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def calculate_metrics(real, predicted):
    mae = mean_absolute_error(
        real,
        predicted
    )
    rmse = np.sqrt(
        mean_squared_error(
            real,
            predicted
        )
    )
    r2 = r2_score(
        real,
        predicted
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }