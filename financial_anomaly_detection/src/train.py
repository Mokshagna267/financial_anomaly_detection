"""Train and persist the anomaly detection model."""
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.config import CONTAMINATION, MODEL_PATH, RANDOM_STATE, RAW_DATA
from src.preprocessing import prepare_features

def train():
    if not RAW_DATA.exists():
        from src.data_generator import generate_transactions
        RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
        generate_transactions().to_csv(RAW_DATA, index=False)

    df = pd.read_csv(RAW_DATA)
    X = prepare_features(df)

    model = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)

    # Evaluation against synthetic ground truth only; real deployments need reviewed labels.
    pred = (model.predict(X) == -1).astype(int)
    score = -model.decision_function(X)
    y = df["ground_truth_anomaly"].astype(int)

    print("\nClassification report:")
    print(classification_report(y, pred, digits=3, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))
    print(f"ROC-AUC: {roc_auc_score(y, score):.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model}, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")

if __name__ == "__main__":
    train()
