from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA = DATA_DIR / "raw" / "transactions.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
DB_PATH = DATA_DIR / "transactions.db"

RANDOM_STATE = 42
CONTAMINATION = 0.03

FEATURES = [
    "amount",
    "transaction_hour",
    "account_age_days",
    "customer_frequency",
    "avg_transaction_amount",
    "distance_from_usual_location",
    "transactions_last_24h",
    "is_online",
    "is_international",
    "merchant_risk",
]
