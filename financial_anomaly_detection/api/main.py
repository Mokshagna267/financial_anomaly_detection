from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.database import init_db, save_prediction, recent_predictions
from src.predictor import AnomalyPredictor

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    init_db()
    predictor = AnomalyPredictor()
    yield

app = FastAPI(
    title="Real-Time Financial Anomaly Detection API",
    version="1.0.0",
    description="Production-style API for real-time transaction anomaly scoring.",
    lifespan=lifespan,
)

class Transaction(BaseModel):
    transaction_id: str = Field(default="LIVE_TXN")
    amount: float = Field(gt=0, le=10_000_000)
    transaction_hour: int = Field(ge=0, le=23)
    merchant_category: Literal[
        "grocery", "restaurant", "electronics", "travel",
        "fuel", "healthcare", "online_services"
    ]
    transaction_type: Literal["card", "online", "bank_transfer", "cash"]
    account_age_days: int = Field(gt=0)
    customer_frequency: int = Field(ge=0)
    avg_transaction_amount: float = Field(gt=0)
    distance_from_usual_location: float = Field(ge=0)
    transactions_last_24h: int = Field(ge=0)
    is_online: int = Field(ge=0, le=1)
    is_international: int = Field(ge=0, le=1)
    merchant_risk: float = Field(ge=0, le=1)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        payload = transaction.model_dump()
        result = predictor.predict(payload)
        save_prediction(payload, result)
        return {
            "transaction_id": payload["transaction_id"],
            **result,
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction could not be completed.")

@app.get("/transactions")
def transactions(limit: int = 100):
    limit = max(1, min(limit, 500))
    return recent_predictions(limit)
