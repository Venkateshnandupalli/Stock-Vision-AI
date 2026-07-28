"""
StockVision AI — Forecast Centre Page (Premium 3D Edition)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from src.utils.config import COMPANY_INFO, ALL_TICKERS

st.set_page_config(page_title="Forecast Centre | StockVision AI", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #030712; color: #e2e8f0; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem !important; max-width: 1440px; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #030712, #060e1c) !important;
        border-right: 1px solid rgba(124,58,237,0.1) !important;
    }

    .page-header {
        background: radial-gradient(ellipse at 70% 30%, rgba(124,58,237,0.15) 0%, transparent 55%),
                    rgba(255,255,255,0.02);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(124,58,237,0.15);
        border-radius: 20px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
    }

    .page-header::before {
        content: '';
        position: absolute; inset: 0;
        background-image:
            linear-gradient(rgba(124,58,237,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(124,58,237,0.03) 1px, transparent 1px);
        background-size: 32px 32px; pointer-events: none;
    }

    .page-title {
        font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #00f5ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        filter: drop-shadow(0 0 12px rgba(124,58,237,0.3));
        margin: 0 0 0.3rem 0;
    }

    .page-subtitle { font-size: 0.85rem; color: #3d5268; }

    .sec-label {
        font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
        color: #e2e8f0; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 8px;
    }
    .sec-line { height: 1px; flex: 1; background: linear-gradient(90deg, rgba(124,58,237,0.4), transparent); }

    /* Neon forecast card */
    .forecast-card {
        background: radial-gradient(ellipse at 50% 0%, rgba(124,58,237,0.12) 0%, transparent 60%),
                    rgba(255,255,255,0.025);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(124,58,237,0.25);
        border-radius: 24px; padding: 3rem 2rem;
        text-align: center; position: relative; overflow: hidden;
        box-shadow: 0 0 60px rgba(124,58,237,0.1), inset 0 1px 0 rgba(255,255,255,0.05);
    }

    .forecast-card::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, #7c3aed, transparent);
    }

    .fc-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #4a6080; margin-bottom: 1rem; }

    .fc-direction-up {
        font-family: 'Syne', sans-serif; font-size: 4rem; font-weight: 800;
        color: #00ff88; text-shadow: 0 0 30px rgba(0,255,136,0.6), 0 0 60px rgba(0,255,136,0.3);
        animation: glowPulse 2s ease-in-out infinite alternate;
    }

    .fc-direction-down {
        font-family: 'Syne', sans-serif; font-size: 4rem; font-weight: 800;
        color: #ff4d6a; text-shadow: 0 0 30px rgba(255,77,106,0.6), 0 0 60px rgba(255,77,106,0.3);
        animation: glowPulse 2s ease-in-out infinite alternate;
    }

    @keyframes glowPulse {
        0%   { filter: brightness(1); }
        100% { filter: brightness(1.25); }
    }

    .fc-return {
        font-family: 'JetBrains Mono', monospace; font-size: 2.5rem; font-weight: 700;
        color: #f8fafc; margin: 0.5rem 0;
    }

    .fc-meta { font-size: 0.75rem; color: #2d4560; margin-top: 1.5rem; font-family: 'JetBrains Mono', monospace; }

    .fc-badge {
        display: inline-block; font-size: 0.65rem; font-weight: 700;
        padding: 4px 12px; border-radius: 100px; margin-top: 1rem;
        text-transform: uppercase; letter-spacing: 1px;
        color: #a78bfa; background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.25);
    }

    /* Info card */
    .info-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 1.6rem;
    }

    .info-table { width: 100%; border-collapse: collapse; }
    .info-table th { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #2d4560; padding: 6px 10px; border-bottom: 1px solid rgba(255,255,255,0.04); }
    .info-table td { font-size: 0.8rem; padding: 10px 10px; border-bottom: 1px solid rgba(255,255,255,0.03); color: #7d8fa8; }
    .info-table td:first-child { font-weight: 600; color: #a78bfa; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; }

    /* Warning bar */
    .warn-bar {
        background: rgba(245,158,11,0.05); border: 1px solid rgba(245,158,11,0.15);
        border-left: 3px solid #f59e0b; border-radius: 10px;
        padding: 0.9rem 1.2rem; font-size: 0.79rem; color: #92702a; line-height: 1.7; margin-top: 1rem;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.025) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important; padding: 1rem !important;
    }
    [data-testid="stMetricLabel"] { color: #4a6080 !important; font-size: 0.72rem !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-family: 'JetBrains Mono', monospace !important; }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#a78bfa,#7c3aed); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown("""
<div class="page-header">
    <div class="page-title">🔮 Forecast Centre</div>
    <div class="page-subtitle">ML-powered return forecasts with 3D accuracy visualization. <strong>Not investment advice — educational analytics only.</strong></div>
</div>
""", unsafe_allow_html=True)

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

st.markdown(f"""
<div class="sec-label">
    Forecast for {company_name} ({ticker})
    <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

# ── Generate prediction ────────────────────────────────────────────────────
col_main, col_info = st.columns([2, 3], gap="large")

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
    pred      = st.session_state["last_prediction"]
    ret       = pred.get("predicted_return_pct", 0)
    direction = pred.get("predicted_direction", "N/A")
    dir_class = "fc-direction-up" if direction == "UP" else "fc-direction-down"
    dir_emoji = "▲" if direction == "UP" else "▼"
    ret_color = "#00ff88" if ret >= 0 else "#ff4d6a"

    with col_main:
        st.markdown(f"""
        <div class="forecast-card">
            <div class="fc-label">{horizon}-Day Return Forecast</div>
            <div class="{dir_class}">{dir_emoji}</div>
            <div class="fc-return" style="color:{ret_color}">{abs(ret):.2f}%</div>
            <div style="font-size:0.85rem;color:#7d8fa8;margin-top:0.3rem">Predicted Direction: <strong style="color:#e2e8f0">{direction}</strong></div>
            <div class="fc-badge">Model: {model_name}</div>
            <div class="fc-meta">Date: {pred.get('prediction_date')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence Gauge
        prob = pred.get("prediction_probability", 0.5)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Confidence Score", "font": {"color": "#4a6080", "size": 13}},
            number={"suffix": "%", "font": {"color": "#f8fafc", "size": 28, "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#2d4560", "tickfont": {"color": "#4a6080"}},
                "bar": {"color": "#7c3aed", "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],  "color": "rgba(255,77,106,0.12)"},
                    {"range": [40, 60], "color": "rgba(245,158,11,0.08)"},
                    {"range": [60, 100],"color": "rgba(0,255,136,0.1)"},
                ],
                "threshold": {"line": {"color": "#00f5ff", "width": 2}, "thickness": 0.75, "value": prob*100},
            },
        ))
        gauge.update_layout(
            height=220, margin=dict(l=20, r=20, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"},
        )
        st.plotly_chart(gauge, use_container_width=True)

with col_info:
    st.markdown("""
    <div class="info-card">
        <div class="sec-label" style="margin-bottom:1rem">ℹ️ How to interpret this forecast <div class="sec-line"></div></div>
        <table class="info-table">
            <thead><tr><th>Field</th><th>Meaning</th></tr></thead>
            <tbody>
                <tr><td>Predicted Return</td><td>Expected % price change over the forecast horizon</td></tr>
                <tr><td>Direction ▲ / ▼</td><td>Predicted price movement direction (UP or DOWN)</td></tr>
                <tr><td>Confidence</td><td>Model probability score for the predicted direction</td></tr>
                <tr><td>Horizon</td><td>1-day = next trading day; 5-day = next week</td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="warn-bar" style="margin-top:1rem">
        ⚠️ <strong>Important:</strong> Stock markets are inherently uncertain. These forecasts
        are derived from historical patterns and statistical models. They should <strong>not</strong>
        be used for actual trading decisions. Model: <code>{model_name}</code> · Horizon: {horizon}d
    </div>
    """, unsafe_allow_html=True)

# ── Historical 3D Accuracy ─────────────────────────────────────────────────
st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec-label">
    📈 Historical Prediction Accuracy — 3D Comparison
    <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

try:
    from src.database.queries import get_latest_predictions
    preds_df = get_latest_predictions(ticker=ticker)
    if not preds_df.empty and "actual_return" in preds_df.columns:
        preds_df = preds_df.dropna(subset=["actual_return", "predicted_return"])
        if not preds_df.empty:
            n         = len(preds_df)
            x_vals    = np.arange(n)
            actual    = preds_df["actual_return"].values
            predicted = preds_df["predicted_return"].values
            errors    = np.abs(actual - predicted)

            # 3D scatter: x=time, y=actual, z=predicted, color=error
            fig3d = go.Figure()
            fig3d.add_trace(go.Scatter3d(
                x=x_vals, y=actual, z=predicted,
                mode="markers+lines",
                marker=dict(
                    size=5,
                    color=errors,
                    colorscale=[[0,"#00ff88"],[0.5,"#f59e0b"],[1,"#ff4d6a"]],
                    showscale=True,
                    colorbar=dict(title="Abs Error", tickfont=dict(color="#4a6080"), titlefont=dict(color="#4a6080")),
                    opacity=0.85,
                    line=dict(color="rgba(0,245,255,0.2)", width=1),
                ),
                line=dict(color="rgba(0,245,255,0.15)", width=1),
                hovertemplate="Actual: %{y:.4f}<br>Predicted: %{z:.4f}<extra></extra>",
            ))

            # Perfect prediction diagonal plane
            diag = np.linspace(min(actual.min(), predicted.min()), max(actual.max(), predicted.max()), 20)
            fig3d.add_trace(go.Surface(
                x=np.outer(np.linspace(0,n,20), np.ones(20)),
                y=np.outer(diag, np.ones(20)),
                z=np.outer(diag, np.ones(20)).T,
                colorscale=[[0,"rgba(0,245,255,0.03)"],[1,"rgba(0,245,255,0.06)"]],
                showscale=False, opacity=0.3, name="Perfect Prediction",
                hoverinfo="skip",
            ))

            fig3d.update_layout(
                scene=dict(
                    xaxis=dict(title="Time", color="#4a6080", showgrid=True, gridcolor="rgba(0,245,255,0.05)"),
                    yaxis=dict(title="Actual Return", color="#4a6080", showgrid=True, gridcolor="rgba(0,245,255,0.05)"),
                    zaxis=dict(title="Predicted Return", color="#4a6080", showgrid=True, gridcolor="rgba(0,245,255,0.05)"),
                    bgcolor="rgba(0,0,0,0)",
                    camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)),
                ),
                height=450, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("No historical predictions with actuals yet. Generate predictions and wait for target dates to pass.")
    else:
        st.info("No stored predictions for this ticker yet.")
except Exception:
    st.info("Connect the database to see historical accuracy.")
