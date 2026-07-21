from fastapi import FastAPI, HTTPException

from app.services import predict_next_close, train_model
from app.schemas import PredictionResponse, TrainRequest

app = FastAPI(
    title="Stock LSTM API",
    description="API para treinamento e previsão do preço de fechamento de ações utilizando LSTM.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "Stock LSTM API",
        "version": "1.0.0"
    }

@app.post(
    "/train",
    tags=["Training"]
)
def train(request: TrainRequest):
    try:
        return train_model(request.stock)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
@app.get(
    "/predict/{stock}",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
def predict(stock: str):
    try:
        return predict_next_close(stock)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get(
    "/health",
    tags=["System"]
)
def health():
    return {"status": "online"}