"""API smoke test; creates the ignored model artifact when needed."""
from fastapi.testclient import TestClient

from api.main import app
from src.config import MODEL_PATH
from src.data_generator import generate_transactions
from src.train import train


def test_health():
    if not MODEL_PATH.exists():
        train()

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "model_loaded": True}
