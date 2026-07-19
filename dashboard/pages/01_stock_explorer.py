"""
StockVision AI — Stock Explorer Page
=======================================
Interactive price & technical indicator explorer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.utils.config import COMPANY_INFO, ALL_TICKERS
from src.database.queries import get_stock_prices, get_technical_indicators

st.set_page_config(page_title="Stock Explorer | StockVision AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f2027 0%, #1a2f3e 100%); }
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Explorer")
st.caption("Analyse historical prices, technical indicators and performance statistics.")

# ── Sidebar controls ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Controls")
    ticker = st.selectbox(
        "Select Stock",
        options=ALL_TICKERS,
        format_func=lambda t: f"{t} — {COMPANY_INFO.get(t, {}).get('name', t)[:25]}",
    )
    days = st.select_slider(
        "Date Range",
        options=[90, 180, 252, 365, 730, 1260],
        value=365,
        format_func=lambda d: f"{d//252}Y {d%252//21}M" if d >= 252 else f"{d}D",
    )
    show_candlestick = st.checkbox("Candlestick chart", value=True)
    show_volume = st.checkbox("Volume bars", value=True)
    show_sma = st.checkbox("Moving averages (SMA 20, 50)", value=True)
    show_bollinger = st.checkbox("Bollinger Bands", value=False)
    show_rsi = st.checkbox("RSI (14)", value=True)

# ── Load data ──────────────────────────────────────────────────────────────
start_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

@st.cache_data(ttl=300)
def load_data(t, start):
    try:
        df = get_stock_prices(t, start_date=start)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.sort_values("trade_date")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_indicators(t, start):
    try:
        df = get_technical_indicators(t, start_date=start)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.sort_values("trade_date")
    except Exception:
        return pd.DataFrame()

df = load_data(ticker, start_date)
indicators = load_indicators(ticker, start_date)
company_name = COMPANY_INFO.get(ticker, {}).get("name", ticker)

if df.empty:
    st.warning(f"No data found for **{ticker}**. Please run the ingestion pipeline first.")
    st.code("python -m src.ingestion.fetch_market_data", language="bash")
    st.stop()

# ── Summary KPI row ────────────────────────────────────────────────────────
st.subheader(f"{company_name} ({ticker})")

latest = df.iloc[-1]
first  = df.iloc[0]
period_return = (latest["close_price"] - first["close_price"]) / first["close_price"] * 100
daily_return  = (df["close_price"].pct_change() * 100).iloc[-1]
high_52w = df["high_price"].max()
low_52w  = df["low_price"].min()
avg_vol  = df["volume"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Close Price",    f"₹{latest['close_price']:,.2f}", f"{daily_return:+.2f}%")
c2.metric("Period Return",  f"{period_return:+.2f}%")
c3.metric("52W High",       f"₹{high_52w:,.2f}")
c4.metric("52W Low",        f"₹{low_52w:,.2f}")
c5.metric("Avg Volume",     f"{avg_vol/1e6:.1f}M")

st.markdown("---")

# ── Price chart ────────────────────────────────────────────────────────────
rows = 2 if (show_volume or show_rsi) else 1
row_heights = [0.6, 0.2, 0.2][:rows] if rows == 2 else [1]
subplot_titles = [f"{ticker} Price"]
if show_volume:
    subplot_titles.append("Volume")
if show_rsi:
    subplot_titles.append("RSI (14)")

fig = make_subplots(
    rows=rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=row_heights[:rows],
)

# Price trace
if show_candlestick:
    fig.add_trace(go.Candlestick(
        x=df["trade_date"],
        open=df["open_price"],
        high=df["high_price"],
        low=df["low_price"],
        close=df["close_price"],
        name="OHLC",
        increasing_line_color="#69f0ae",
        decreasing_line_color="#ff5252",
    ), row=1, col=1)
else:
    fig.add_trace(go.Scatter(
        x=df["trade_date"], y=df["close_price"],
        mode="lines", name="Close",
        line=dict(color="#4fc3f7", width=2),
    ), row=1, col=1)

# Moving averages
if show_sma and not indicators.empty:
    for col, color, label in [
        ("sma_20", "#ffab40", "SMA 20"),
        ("sma_50", "#ce93d8", "SMA 50"),
    ]:
        if col in indicators.columns:
            fig.add_trace(go.Scatter(
                x=indicators["trade_date"], y=indicators[col],
                mode="lines", name=label,
                line=dict(color=color, width=1.5, dash="dot"),
            ), row=1, col=1)

# Bollinger Bands
if show_bollinger and not indicators.empty:
    if "bollinger_upper" in indicators.columns:
        fig.add_trace(go.Scatter(
            x=indicators["trade_date"], y=indicators["bollinger_upper"],
            name="BB Upper", line=dict(color="rgba(255,255,255,0.3)", width=1),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=indicators["trade_date"], y=indicators["bollinger_lower"],
            name="BB Lower", line=dict(color="rgba(255,255,255,0.3)", width=1),
            fill="tonexty", fillcolor="rgba(255,255,255,0.04)",
        ), row=1, col=1)

# Volume
if show_volume:
    colors = ["#69f0ae" if r >= 0 else "#ff5252"
              for r in df["close_price"].pct_change().fillna(0)]
    fig.add_trace(go.Bar(
        x=df["trade_date"], y=df["volume"],
        name="Volume", marker_color=colors, opacity=0.6,
    ), row=2, col=1)

# RSI
if show_rsi and not indicators.empty and "rsi_14" in indicators.columns:
    rsi_row = 2 if show_volume else 2
    fig.add_trace(go.Scatter(
        x=indicators["trade_date"], y=indicators["rsi_14"],
        name="RSI 14", line=dict(color="#ff9800", width=1.5),
    ), row=rsi_row, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   opacity=0.5, row=rsi_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=rsi_row, col=1)

fig.update_layout(
    template="plotly_dark",
    height=550,
    xaxis_rangeslider_visible=False,
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(fig, use_container_width=True)

# ── Statistics table ───────────────────────────────────────────────────────
st.subheader("📐 Summary Statistics")
daily_ret = df["close_price"].pct_change().dropna()
stats = {
    "Trading Days":         len(df),
    "Latest Close (₹)":    f"₹{latest['close_price']:,.2f}",
    "Period Return":        f"{period_return:+.2f}%",
    "Daily Return (mean)":  f"{daily_ret.mean()*100:.4f}%",
    "Annualized Return":    f"{daily_ret.mean()*252*100:.2f}%",
    "Annualized Volatility": f"{daily_ret.std()*252**0.5*100:.2f}%",
    "Max Daily Gain":       f"{daily_ret.max()*100:.2f}%",
    "Max Daily Loss":       f"{daily_ret.min()*100:.2f}%",
    "Positive Days":        f"{(daily_ret>0).mean()*100:.1f}%",
    "52W High":             f"₹{high_52w:,.2f}",
    "52W Low":              f"₹{low_52w:,.2f}",
    "Avg Volume":           f"{avg_vol/1e6:.2f}M",
}
stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
st.dataframe(stats_df, hide_index=True, use_container_width=True)
