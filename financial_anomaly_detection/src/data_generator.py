"""Generate realistic synthetic financial transactions with injected anomalies."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import RAW_DATA, RANDOM_STATE

RNG = np.random.default_rng(RANDOM_STATE)

MERCHANTS = {
    "grocery": 0.10,
    "restaurant": 0.15,
    "electronics": 0.55,
    "travel": 0.70,
    "fuel": 0.20,
    "healthcare": 0.15,
    "online_services": 0.40,
}
TYPES = {"card": 0, "online": 1, "bank_transfer": 2, "cash": 3}

def generate_transactions(n=12000, anomaly_rate=0.03):
    n_anom = int(n * anomaly_rate)
    n_normal = n - n_anom

    customer = RNG.integers(10000, 18000, n)
    amount = np.clip(RNG.lognormal(mean=3.6, sigma=0.75, size=n), 5, 25000)
    hour = RNG.integers(6, 23, n)
    merchant = RNG.choice(list(MERCHANTS), n)
    transaction_type = RNG.choice(list(TYPES), n, p=[0.55, 0.28, 0.12, 0.05])
    account_age = RNG.integers(30, 2500, n)
    frequency = RNG.poisson(7, n) + 1
    avg_amount = np.clip(amount * RNG.normal(0.75, 0.18, n), 20, None)
    distance = np.clip(RNG.exponential(12, n), 0, 250)
    last24 = RNG.poisson(2.5, n)
    international = RNG.binomial(1, 0.06, n)

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:07d}" for i in range(n)],
        "customer_id": customer,
        "amount": amount.round(2),
        "transaction_hour": hour,
        "merchant_category": merchant,
        "transaction_type": transaction_type,
        "account_age_days": account_age,
        "customer_frequency": frequency,
        "avg_transaction_amount": avg_amount.round(2),
        "distance_from_usual_location": distance.round(2),
        "transactions_last_24h": last24,
        "is_online": (transaction_type == "online").astype(int),
        "is_international": international,
        "merchant_risk": [MERCHANTS[m] for m in merchant],
        "ground_truth_anomaly": 0,
    })

    # Inject deliberately suspicious behavior.
    idx = RNG.choice(df.index, n_anom, replace=False)
    df.loc[idx, "amount"] = RNG.uniform(30000, 150000, n_anom).round(2)
    df.loc[idx, "transaction_hour"] = RNG.choice([0, 1, 2, 3, 4, 5], n_anom)
    df.loc[idx, "distance_from_usual_location"] = RNG.uniform(500, 5000, n_anom).round(2)
    df.loc[idx, "transactions_last_24h"] = RNG.integers(15, 50, n_anom)
    df.loc[idx, "is_international"] = 1
    df.loc[idx, "merchant_category"] = RNG.choice(["electronics", "travel", "online_services"], n_anom)
    df.loc[idx, "merchant_risk"] = df.loc[idx, "merchant_category"].map(MERCHANTS)
    df.loc[idx, "ground_truth_anomaly"] = 1

    return df

if __name__ == "__main__":
    RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    df = generate_transactions()
    df.to_csv(RAW_DATA, index=False)
    print(f"Generated {len(df):,} transactions -> {RAW_DATA}")
    print(f"Injected anomalies: {df.ground_truth_anomaly.sum():,}")
