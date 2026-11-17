import sqlite3
from datetime import datetime, timezone
from src.config import DB_PATH

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            amount REAL,
            merchant_category TEXT,
            transaction_type TEXT,
            is_anomaly INTEGER,
            anomaly_score REAL,
            risk_score REAL,
            risk_level TEXT,
            created_at TEXT
        )
        """)
        conn.commit()

def save_prediction(txn: dict, result: dict):
    with get_connection() as conn:
        conn.execute("""
        INSERT INTO predictions
        (transaction_id, amount, merchant_category, transaction_type,
         is_anomaly, anomaly_score, risk_score, risk_level, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            txn.get("transaction_id"),
            txn["amount"],
            txn.get("merchant_category"),
            txn.get("transaction_type"),
            int(result["is_anomaly"]),
            result["anomaly_score"],
            result["risk_score"],
            result["risk_level"],
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()

def recent_predictions(limit=100):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
