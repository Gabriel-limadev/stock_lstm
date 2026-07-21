from pydantic import BaseModel


class TrainRequest(BaseModel):
    stock: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "stock": "PETR4.SA"
            }
        }
    }


class PredictionResponse(BaseModel):
    stock: str
    prediction_date: str
    last_close: float
    predicted_close: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "stock": "PETR4.SA",
                "prediction_date": "2026-07-20",
                "last_close": 32.85,
                "predicted_close": 33.12
            }
        }
    }