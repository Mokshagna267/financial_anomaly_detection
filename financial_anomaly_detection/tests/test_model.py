from src.data_generator import generate_transactions
from src.preprocessing import prepare_features

def test_generated_dataset_has_required_features():
    df = generate_transactions(100)
    X = prepare_features(df)
    assert len(X) == 100
    assert X.shape[1] == 10

def test_anomalies_are_injected():
    df = generate_transactions(1000, anomaly_rate=0.05)
    assert int(df["ground_truth_anomaly"].sum()) == 50
