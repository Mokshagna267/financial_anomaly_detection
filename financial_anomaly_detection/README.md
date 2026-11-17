# Real-Time Financial Transaction Anomaly Detection

An end-to-end machine-learning application that identifies potentially unusual financial transactions. It combines an unsupervised **Isolation Forest** model, a **FastAPI** inference service, **SQLite** prediction history, and a **Streamlit** monitoring dashboard.

> **Important:** This is an educational portfolio project using synthetic data. It is not a production fraud-detection system and must not be used to make real financial decisions.

## Architecture

```text
Synthetic transaction data
          |
          v
   Feature preparation ---> Isolation Forest model
                                  |
                                  v
Client ---> FastAPI /predict ---> Risk score
                                  |
                                  v
                              SQLite DB
                                  |
                                  v
                         Streamlit dashboard
```

## Features

- Synthetic transaction generator with configurable anomaly injection.
- Unsupervised anomaly detection using scikit-learn Isolation Forest.
- REST API with Pydantic request validation and interactive OpenAPI documentation.
- Bounded 0–100 risk score and LOW / MEDIUM / HIGH risk labels.
- SQLite persistence and endpoint for recent predictions.
- Streamlit dashboard with summary metrics and transaction charts.
- Automated tests and Docker Compose setup.

## Tech stack

Python 3.11+, NumPy, pandas, scikit-learn, FastAPI, Pydantic, SQLite, Streamlit, pytest, Docker.

## Run locally

### 1. Clone and enter the repository

```bash
git clone https://github.com/<your-username>/financial-anomaly-detection.git
cd financial-anomaly-detection
```

### 2. Create a virtual environment

```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies and prepare the project

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

`main.py` generates the synthetic dataset, trains and saves the model under `models/`, and initializes the SQLite database under `data/`. Generated data, model binaries, and database files are intentionally excluded from Git; they are recreated locally.

### 4. Start the API

```bash
uvicorn api.main:app --reload
```

- API base: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 5. Start the dashboard

Open a second terminal, activate the same virtual environment, and run:

```bash
streamlit run dashboard/app.py
```

Dashboard: http://localhost:8501

## Try a prediction

Send a POST request to `/predict` with a JSON transaction. Example using `curl` (works in macOS/Linux and Git Bash):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"LIVE001","amount":85000,"transaction_hour":2,"merchant_category":"electronics","transaction_type":"online","account_age_days":120,"customer_frequency":18,"avg_transaction_amount":4200,"distance_from_usual_location":850,"transactions_last_24h":14,"is_online":1,"is_international":1,"merchant_risk":0.55}'
```

To inspect stored predictions, open http://localhost:8000/transactions. The exact anomaly result and risk score depend on the trained model.

## Run tests

```bash
python -m pytest -q
```

The API smoke test trains the model automatically if the ignored model file is not present.

## Run with Docker Compose

With Docker Desktop / Docker Engine and the Compose plugin installed:

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

The API service prepares the model and database on startup. Compose configures the dashboard to reach the API using the internal service hostname.

## Project structure

```text
financial-anomaly-detection/
├── api/                 # FastAPI routes and request schema
├── dashboard/           # Streamlit monitoring dashboard
├── data/raw/            # Generated dataset (not committed)
├── models/              # Trained model artifact (not committed)
├── src/                 # Data generation, preprocessing, training, prediction, DB
├── tests/               # Automated tests
├── Dockerfile
├── docker-compose.yml
├── main.py              # One-command data/model/database setup
├── requirements.txt
└── README.md
```

## Model and evaluation notes

The Isolation Forest is trained without using the synthetic `ground_truth_anomaly` column. That column is used only to print demonstration evaluation metrics after fitting. Because the evaluation is performed on the same synthetic dataset used for training, those metrics are **not** a reliable estimate of performance on unseen transactions. Synthetic patterns are deliberately injected and may be easier to detect than real-world fraud.

A real deployment would require representative and reviewed labels, time-aware validation, threshold calibration, drift monitoring, explainability, access controls, privacy safeguards, and a human investigation process.

## Future improvements

- Add a held-out, time-based evaluation dataset.
- Add model/version metadata and configurable decision thresholds.
- Add structured logging, API authentication, and rate limiting.
- Add data-drift and model-performance monitoring.

## Resume description

**Real-Time Financial Transaction Anomaly Detection | Python, Scikit-learn, FastAPI, Streamlit, SQLite, Docker**

- Built an end-to-end unsupervised anomaly-detection pipeline using Isolation Forest to identify unusual synthetic financial transactions.
- Developed a FastAPI service with Pydantic validation, risk scoring, and SQLite prediction logging.
- Created a Streamlit dashboard for monitoring transaction volume, anomaly flags, risk levels, and prediction history; added automated tests and Docker Compose deployment.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
