"""
StockVision AI — Streamlit Dashboard
======================================
Multi-page interactive data application.
Pages:
  1. Home — Market overview and latest status
  2. Stock Explorer — Price charts, indicators (pages/01_stock_explorer.py)
  3. Forecast Centre — Model predictions (pages/02_forecast_centre.py)
  4. Model Laboratory — Leaderboard, feature importance (pages/03_model_lab.py)
  5. Data Quality — Pipeline health, missing data (pages/04_data_quality.py)

Run:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# ── Add project root to path ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from src.utils.config import COMPANY_INFO, BENCHMARK_TICKER, BENCHMARK_NAME, ALL_TICKERS
from src.database.queries import (
    get_stock_prices,
    get_all_tickers,
    get_latest_predictions,
    get_model_metrics,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/your-repo/stockvision-ai",
        "About": "StockVision AI — Market Analytics & Forecasting Platform",
    },
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }

    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 1rem;
        color: rgba(255,255,255,0.75);
        margin-top: 0.4rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #1e2a38, #263445);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4fc3f7;
    }

    .metric-value.positive { color: #69f0ae; }
    .metric-value.negative { color: #ff5252; }

    .metric-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.55);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 0.3rem;
    }

    .disclaimer {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        font-size: 0.82rem;
        color: #ffc107;
        margin-top: 1rem;
    }

    .stSelectbox > div > div {
        border-radius: 8px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #1a2f3e 100%);
    }

    [data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.85) !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_prices(ticker: str, days: int = 365) -> pd.DataFrame:
    """Load recent price data with caching."""
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        df = get_stock_prices(ticker, start_date=start)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.sort_values("trade_date")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    try:
        return get_latest_predictions()
    except Exception:
        return pd.DataFrame()


def format_return(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    sign = "▲" if value > 0 else "▼"
    return f"{sign} {abs(value):.2f}%"


def get_return_class(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return "positive" if value > 0 else "negative"


# ── Home Page ──────────────────────────────────────────────────────────────

def render_home():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 StockVision AI</h1>
        <p>Market Analytics, Risk Intelligence & Price Forecasting Platform · NIFTY 50 Universe</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Last Updated:** {datetime.now().strftime('%d %b %Y, %I:%M %p IST')}")

    # Market overview ticker row
    st.subheader("🗂️ Market Snapshot")

    tickers_to_show = ALL_TICKERS[:6]
    cols = st.columns(len(tickers_to_show))

    for col, ticker in zip(cols, tickers_to_show):
        df = load_prices(ticker, days=5)
        if df.empty or len(df) < 2:
            col.metric(label=ticker, value="N/A")
            continue

        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        change = (latest["close_price"] - prev["close_price"]) / prev["close_price"] * 100
        name = COMPANY_INFO.get(ticker, {}).get("name", ticker)[:15]

        col.metric(
            label=f"{name}",
            value=f"₹{latest['close_price']:,.2f}",
            delta=f"{change:+.2f}%",
        )

    st.markdown("---")

    # Two-column layout
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Price chart for NIFTY 50
        st.subheader(f"📊 {BENCHMARK_NAME} — 1 Year Performance")
        nifty_df = load_prices(BENCHMARK_TICKER, days=365)
        if not nifty_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=nifty_df["trade_date"],
                y=nifty_df["close_price"],
                mode="lines",
                name="NIFTY 50",
                line=dict(color="#4fc3f7", width=2),
                fill="tozeroy",
                fillcolor="rgba(79, 195, 247, 0.08)",
            ))
            fig.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="",
                yaxis_title="Index Value",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the ingestion pipeline to load market data.")

    with col_right:
        # Sector performance bar
        st.subheader("🏭 Sector Distribution")
        sectors = {}
        for ticker, info in COMPANY_INFO.items():
            sec = info.get("sector", "Other")
            if sec != "Benchmark":
                sectors[sec] = sectors.get(sec, 0) + 1

        sec_df = pd.DataFrame(list(sectors.items()), columns=["Sector", "Stocks"])
        fig = px.bar(
            sec_df,
            x="Stocks",
            y="Sector",
            orientation="h",
            color="Stocks",
            color_continuous_scale="blues",
            template="plotly_dark",
        )
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Latest predictions mini-table
        st.subheader("🔮 Latest Forecasts")
        preds_df = load_predictions()
        if not preds_df.empty:
            display_cols = ["ticker", "model_name", "predicted_direction", "prediction_probability"]
            existing = [c for c in display_cols if c in preds_df.columns]
            st.dataframe(
                preds_df[existing].head(8).rename(columns={
                    "ticker": "Ticker",
                    "model_name": "Model",
                    "predicted_direction": "Direction",
                    "prediction_probability": "Confidence",
                }),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Run training and prediction pipeline to see forecasts.")

    # KPI Summary
    st.markdown("---")
    st.subheader("📐 Platform Summary")
    kpi_cols = st.columns(4)
    kpi_data = [
        ("Stocks Tracked",  str(len(ALL_TICKERS)), ""),
        ("Sectors Covered", "4",                  "IT · Banking · Energy · Auto"),
        ("Benchmark",       BENCHMARK_NAME,        "Reference index"),
        ("Data Source",     "yfinance",             "5+ years daily OHLCV"),
    ]
    for col, (label, value, sub) in zip(kpi_cols, kpi_data):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            <div style="font-size:0.72rem;color:rgba(255,255,255,0.4);margin-top:4px">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> All forecasts are generated from historical data for
        analytical and educational purposes only. They do not constitute investment advice.
        Past market patterns may not predict future performance.
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 StockVision AI")
    st.markdown("---")
    st.markdown("""
    **Navigation**

    Use the pages in the sidebar to explore:
    - 📊 Stock Explorer
    - 🔮 Forecast Centre
    - 🧪 Model Laboratory
    - 🔍 Data Quality
    """)
    st.markdown("---")
    st.markdown("**Universe:** NIFTY 50 (10 stocks)")
    st.markdown("**Forecast:** Next-day & 5-day return")
    st.markdown("**Models:** XGBoost, RF, Logistic")
    st.markdown("---")
    st.caption("Built for Data Analyst Portfolio · 2024")

# ── Render main page ───────────────────────────────────────────────────────
render_home()
