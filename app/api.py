from fastapi import FastAPI, HTTPException, Request
from app.services import predict_next_close, train_model
from app.schemas import PredictionResponse, TrainRequest

import logging
import psutil
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock LSTM API",
    description="API para treinamento e previsão do preço de fechamento de ações utilizando LSTM.",
    version="1.0.0"
)


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start

    cpu = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory().percent

    logger.info(
        f"{request.client.host} | "
        f"{request.method} {request.url.path} | "
        f"Status={response.status_code} | "
        f"Tempo={duration:.3f}s | "
        f"CPU={cpu}% | "
        f"RAM={memory}%"
    )

    return response


@app.get("/")
def root():
    return {
        "title": "Stock LSTM API",
        "description": "API para treinamento e previsão do preço de fechamento de ações brasileiras utilizando LSTM.",
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
        logger.warning(str(e))
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception:
        logger.exception("Erro durante o treinamento.")
        raise HTTPException(
            status_code=500,
            detail="Erro interno durante o treinamento."
        )


@app.get(
    "/predict/{stock}",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
def predict(stock: str):
    try:
        return predict_next_close(stock)

    except ValueError as e:
        logger.warning(str(e))
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception:
        logger.exception("Erro durante a previsão.")
        raise HTTPException(
            status_code=500,
            detail="Erro interno durante a previsão."
        )


@app.get(
    "/health",
    tags=["System"]
)
def health():
    return {
        "status": "online",
        "version": app.version,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": psutil.virtual_memory().percent
    }