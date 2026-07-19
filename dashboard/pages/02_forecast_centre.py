"""
StockVision AI — Forecast Centre Page
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from src.utils.config import COMPANY_INFO, ALL_TICKERS

st.set_page_config(page_title="Forecast Centre | StockVision AI", page_icon="🔮", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f2027 0%, #1a2f3e 100%); }
.forecast-card {
    background: linear-gradient(145deg, #1a2f3e, #0f2027);
    border: 1px solid rgba(79, 195, 247, 0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.direction-up   { color: #69f0ae; font-size: 3rem; font-weight: 700; }
.direction-down { color: #ff5252; font-size: 3rem; font-weight: 700; }
.confidence-bar { margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Forecast Centre")
st.caption(
    "Model-generated return forecasts and direction predictions. "
    "**Not investment advice — educational analytics only.**"
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Forecast Settings")
    ticker = st.selectbox(
        "Select Stock",
        options=ALL_TICKERS,
        format_func=lambda t: f"{t} — {COMPANY_INFO.get(t, {}).get('name', t)[:25]}",
    )
    model_name = st.selectbox(
        "Select Model",
        options=["xgboost_regressor", "random_forest", "linear_regression",
                 "gradient_boosting", "ridge_regression"],
    )
    horizon = st.radio("Forecast Horizon", options=[1, 5], format_func=lambda h: f"{h}-day")

company_name = COMPANY_INFO.get(ticker, {}).get("name", ticker)
st.subheader(f"Forecast for **{company_name}** ({ticker})")

# ── Generate prediction ────────────────────────────────────────────────────
col_main, col_info = st.columns([2, 3])

with col_main:
    if st.button("⚡ Generate Forecast", use_container_width=True, type="primary"):
        try:
            from src.models.predict import predict_ticker
            with st.spinner("Running model inference..."):
                result = predict_ticker(
                    ticker, model_name=model_name, horizon=horizon,
                    task="regression", save_to_db=False,
                )
            st.session_state["last_prediction"] = result
            st.success("Forecast generated!")
        except FileNotFoundError:
            st.error(
                f"⚠️ Model **{model_name}** not found for **{ticker}**. "
                "Please run the training pipeline first:\n"
                "```\npython -m src.models.train --ticker " + ticker + "\n```"
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

if "last_prediction" in st.session_state:
    pred = st.session_state["last_prediction"]
    ret  = pred.get("predicted_return_pct", 0)
    direction = pred.get("predicted_direction", "N/A")
    dir_class = "direction-up" if direction == "UP" else "direction-down"
    dir_emoji = "▲" if direction == "UP" else "▼"

    with col_main:
        st.markdown(f"""
        <div class="forecast-card">
            <div class="metric-label" style="color:rgba(255,255,255,0.5);margin-bottom:0.5rem">
                {horizon}-Day Return Forecast
            </div>
            <div class="{dir_class}">{dir_emoji} {abs(ret):.2f}%</div>
            <div style="color:rgba(255,255,255,0.6);margin-top:0.5rem;font-size:0.9rem">
                Predicted Direction: <strong>{direction}</strong>
            </div>
            <div style="font-size:0.78rem;color:rgba(255,255,255,0.35);margin-top:1.5rem">
                Model: {model_name} · Date: {pred.get('prediction_date')}
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_info:
    st.markdown("### ℹ️ How to interpret this forecast")
    st.markdown("""
    | Field | Meaning |
    |---|---|
    | **Predicted Return** | Expected % price change over the horizon |
    | **Direction ▲/▼** | Predicted price movement direction |
    | **Horizon** | 1-day = next trading day; 5-day = next week |

    > **Important:** Stock markets are inherently uncertain.
    > These forecasts are derived from historical patterns and statistical models.
    > They should **not** be used for actual trading decisions.

    #### Model used: `{model}`
    """.format(model=model_name))

# ── Historical prediction accuracy ────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Historical Prediction Accuracy")

try:
    from src.database.queries import get_latest_predictions
    preds_df = get_latest_predictions(ticker=ticker)
    if not preds_df.empty and "actual_return" in preds_df.columns:
        preds_df = preds_df.dropna(subset=["actual_return", "predicted_return"])
        if not preds_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=preds_df["prediction_date"],
                y=preds_df["actual_return"],
                name="Actual Return",
                line=dict(color="#69f0ae", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=preds_df["prediction_date"],
                y=preds_df["predicted_return"],
                name="Predicted Return",
                line=dict(color="#4fc3f7", width=1.5, dash="dash"),
            ))
            fig.update_layout(
                template="plotly_dark", height=300,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="Date", yaxis_title="Return (%)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical predictions with actuals yet. Generate predictions and wait for target dates to pass.")
    else:
        st.info("No stored predictions for this ticker yet.")
except Exception:
    st.info("Connect the database to see historical accuracy.")

# ── Disclaimer ─────────────────────────────────────────────────────────────
st.markdown("""
> ⚠️ **Disclaimer:** Forecasts are generated from historical market data for **analytical and
> educational purposes only**. They do not constitute financial advice. Past patterns do
> not guarantee future results. Always consult a qualified financial advisor before making investment decisions.
""")
