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
    last_available_date: str
    prediction_date: str
    last_close: float
    predicted_close: float