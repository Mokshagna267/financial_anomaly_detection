"""One-command local setup: generate data, train model, initialize DB."""
from src.data_generator import generate_transactions
from src.config import RAW_DATA
from src.train import train
from src.database import init_db

if __name__ == "__main__":
    RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    generate_transactions().to_csv(RAW_DATA, index=False)
    print(f"Dataset created: {RAW_DATA}")
    train()
    init_db()
    print("Setup complete.")
