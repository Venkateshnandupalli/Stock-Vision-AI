"""
StockVision AI — Model Laboratory Page
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Model Lab | StockVision AI", page_icon="🧪", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f2027 0%, #1a2f3e 100%); }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Model Laboratory")
st.caption("Model comparison leaderboard, feature importance, and validation results.")

# ── Load metrics ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_metrics():
    try:
        from src.database.queries import get_model_metrics
        return get_model_metrics()
    except Exception:
        return pd.DataFrame()

metrics_df = load_metrics()

if metrics_df.empty:
    st.info(
        "No model metrics found. Run training to populate the leaderboard:\n"
        "```\npython -m src.models.train\n```"
    )

    # Show placeholder demo table
    st.subheader("📋 Model Leaderboard (Demo)")
    demo = pd.DataFrame({
        "Model":      ["XGBoost Regressor", "Random Forest", "Gradient Boosting", "Ridge Regression", "Naive Baseline"],
        "MAE":        [0.0082, 0.0091, 0.0089, 0.0098, 0.0110],
        "RMSE":       [0.0121, 0.0134, 0.0129, 0.0142, 0.0158],
        "Dir Acc.":   ["54.2%", "52.8%", "53.1%", "51.4%", "50.0%"],
        "vs Baseline":["+25.5%", "+17.3%", "+19.1%", "+10.9%", "Baseline"],
    })
    st.dataframe(demo, hide_index=True, use_container_width=True)
    st.stop()

# ── Leaderboard ────────────────────────────────────────────────────────────
st.subheader("📋 Model Leaderboard")

col_filter1, col_filter2 = st.columns(2)
with col_filter1:
    if "ticker" in metrics_df.columns:
        tickers = ["All"] + list(metrics_df["ticker"].dropna().unique())
        selected_ticker = st.selectbox("Filter by Ticker", tickers)
    else:
        selected_ticker = "All"

with col_filter2:
    if "target" in metrics_df.columns:
        targets = ["All"] + list(metrics_df["target"].dropna().unique())
        selected_target = st.selectbox("Filter by Target", targets)
    else:
        selected_target = "All"

filtered = metrics_df.copy()
if selected_ticker != "All" and "ticker" in filtered.columns:
    filtered = filtered[filtered["ticker"] == selected_ticker]
if selected_target != "All" and "target" in filtered.columns:
    filtered = filtered[filtered["target"] == selected_target]

display_cols = [
    "model_name", "ticker", "target", "mae", "rmse", "r_squared",
    "directional_accuracy", "f1_score", "roc_auc",
    "baseline_mae", "improvement_over_baseline",
]
existing_cols = [c for c in display_cols if c in filtered.columns]
st.dataframe(filtered[existing_cols].sort_values("mae"), hide_index=True, use_container_width=True)

st.markdown("---")

# ── MAE comparison chart ───────────────────────────────────────────────────
if "model_name" in filtered.columns and "mae" in filtered.columns:
    st.subheader("📊 MAE by Model")
    mae_agg = filtered.groupby("model_name")["mae"].mean().reset_index().sort_values("mae")
    fig = px.bar(
        mae_agg, x="mae", y="model_name", orientation="h",
        color="mae", color_continuous_scale="blues_r",
        template="plotly_dark",
        labels={"mae": "Mean Absolute Error", "model_name": "Model"},
    )
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0),
                      coloraxis_showscale=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# ── Directional accuracy ───────────────────────────────────────────────────
if "directional_accuracy" in filtered.columns:
    st.subheader("🎯 Directional Accuracy by Model")
    da_agg = filtered.groupby("model_name")["directional_accuracy"].mean().reset_index()
    fig2 = px.bar(
        da_agg, x="model_name", y="directional_accuracy",
        color="directional_accuracy", color_continuous_scale="greens",
        template="plotly_dark",
        labels={"directional_accuracy": "Directional Accuracy", "model_name": "Model"},
    )
    fig2.add_hline(y=0.5, line_dash="dash", line_color="red",
                   annotation_text="Random Baseline (50%)", opacity=0.7)
    fig2.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0),
                       coloraxis_showscale=False,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

# ── Interpretation note ────────────────────────────────────────────────────
st.markdown("""
> **Interpretation guide:**
> - **MAE**: Mean Absolute Error — lower is better
> - **Directional Accuracy**: % of times the model correctly predicted up/down direction
> - **vs Baseline**: % improvement over naive random-walk baseline (predict yesterday's return)
> - A directional accuracy of ~55%+ is considered meaningful for financial time series
""")
