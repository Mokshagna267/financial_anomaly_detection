import pandas as pd
from src.config import FEATURES

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw transaction records into the model's numeric feature matrix."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    X = df[FEATURES].copy()
    return X.replace([float("inf"), float("-inf")], 0).fillna(0)
