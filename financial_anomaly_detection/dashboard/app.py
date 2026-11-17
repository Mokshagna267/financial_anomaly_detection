import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Financial Anomaly Monitor",
    page_icon="🚨",
    layout="wide",
)

API_URL = st.sidebar.text_input("API URL", os.getenv("API_URL", "http://localhost:8000"))

st.title("🚨 Real-Time Financial Anomaly Monitor")
st.caption("Isolation Forest + FastAPI + SQLite | Synthetic data for demonstration")

def load_transactions():
    try:
        r = requests.get(f"{API_URL}/transactions?limit=500", timeout=3)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()

df = load_transactions()

c1, c2, c3, c4 = st.columns(4)
total = len(df)
anomalies = int(df["is_anomaly"].sum()) if total else 0
high = int((df["risk_level"] == "HIGH").sum()) if total else 0
rate = (anomalies / total * 100) if total else 0

c1.metric("Transactions", total)
c2.metric("Anomalies", anomalies)
c3.metric("High Risk", high)
c4.metric("Anomaly Rate", f"{rate:.2f}%")

st.divider()

if df.empty:
    st.info("No transactions yet. Start the API and send predictions.")
else:
    left, right = st.columns(2)
    with left:
        st.subheader("Risk Distribution")
        st.bar_chart(df["risk_level"].value_counts())
    with right:
        st.subheader("Transaction Amounts")
        chart = df[["created_at", "amount"]].copy()
        chart["created_at"] = pd.to_datetime(chart["created_at"])
        chart = chart.set_index("created_at").sort_index()
        st.line_chart(chart)

    st.subheader("Recent Transactions")
    display_cols = [
        "transaction_id", "amount", "merchant_category",
        "transaction_type", "is_anomaly", "risk_score",
        "risk_level", "created_at"
    ]
    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
    )

if st.sidebar.button("Refresh"):
    st.rerun()
