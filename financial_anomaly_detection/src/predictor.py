import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.preprocessing import prepare_features

class AnomalyPredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run: python -m src.data_generator && python -m src.train"
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]

    def predict(self, transaction: dict) -> dict:
        df = pd.DataFrame([transaction])
        X = prepare_features(df)
        raw_score = float(-self.model.decision_function(X)[0])
        prediction = int(self.model.predict(X)[0] == -1)

        # Convert model score into a UI-friendly 0-100 risk score.
        risk_score = max(0.0, min(100.0, 50 + raw_score * 100))
        if prediction:
            risk_level = "HIGH" if risk_score >= 70 else "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "is_anomaly": bool(prediction),
            "anomaly_score": round(raw_score, 4),
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
        }
