"""
StockVision AI — Stock Explorer Page (Premium 3D Edition)
==========================================================
Interactive price & technical indicator explorer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.utils.config import COMPANY_INFO, ALL_TICKERS
from src.database.queries import get_stock_prices, get_technical_indicators

st.set_page_config(page_title="Stock Explorer | StockVision AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Space+Grotesk:wght@400;500;600;700');

html, body { background: #030712 !important; }
[data-testid="stApp"], [data-testid="stAppViewContainer"] { background: #030712 !important; }
[data-testid="stAppViewBlockContainer"] { animation: pageFadeIn 0.35s ease; }
@keyframes pageFadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background: #030712; color: #e2e8f0; }
#MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden !important; height: 0 !important; }
    .stDeployButton { display: none !important; }
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stTopBar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stMainBlockContainer"], [data-testid="stMain"],
    section[data-testid="stSidebar"] ~ div, .main .block-container { background: #030712 !important; }
.block-container { padding-top: 1rem !important; max-width: 1440px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #030712, #060e1c) !important;
    border-right: 1px solid rgba(0,245,255,0.08) !important;
}

[data-testid="stSidebarNav"] a {
    display: flex !important; align-items: center !important;
    padding: 9px 12px !important; border-radius: 10px !important;
    font-size: 0.84rem !important; font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #4a6080 !important; text-decoration: none !important;
    transition: all 0.22s ease !important; border: 1px solid transparent !important;
    margin-bottom: 2px !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(0,245,255,0.06) !important;
    border-color: rgba(0,245,255,0.12) !important;
    color: #00f5ff !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(0,245,255,0.09) !important;
    border-color: rgba(0,245,255,0.18) !important;
    color: #00f5ff !important; font-weight: 600 !important;
    box-shadow: 0 0 18px rgba(0,245,255,0.07) !important;
}
[data-testid="stSidebarNav"] span { color: inherit !important; font-weight: inherit !important; }

    .page-header {
        background: radial-gradient(ellipse at 80% 30%, rgba(0,245,255,0.08) 0%, transparent 55%),
                    rgba(255,255,255,0.02);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0,245,255,0.1);
        border-radius: 20px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
    }

    .page-header::before {
        content: '';
        position: absolute; inset: 0;
        background-image:
            linear-gradient(rgba(0,245,255,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,245,255,0.025) 1px, transparent 1px);
        background-size: 32px 32px;
        pointer-events: none;
    }

    .page-title {
        font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 800;
        color: #f8fafc; margin: 0 0 0.3rem 0;
        background: linear-gradient(90deg, #00f5ff, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        filter: drop-shadow(0 0 12px rgba(0,245,255,0.25));
    }

    .page-subtitle { font-size: 0.85rem; color: #3d5268; }

    .kpi-glass {
        background: rgba(255,255,255,0.025);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 1.2rem 1.4rem;
        transition: all 0.3s ease; position: relative;
    }

    .kpi-glass:hover {
        border-color: rgba(0,245,255,0.2);
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 20px rgba(0,245,255,0.05);
    }

    .sec-label {
        font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700;
        color: #e2e8f0; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 8px;
    }

    .sec-line { height: 1px; flex: 1; background: linear-gradient(90deg, rgba(0,245,255,0.3), transparent); }

    .stats-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 1.4rem 1.6rem;
    }

    .stat-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.83rem;
    }

    .stat-row:last-child { border-bottom: none; }
    .stat-metric { color: #4a6080; }
    .stat-value  { font-family: 'JetBrains Mono', monospace; color: #e2e8f0; font-weight: 600; }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.025) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important; padding: 1rem 1.2rem !important;
    }
    [data-testid="stMetricLabel"] { color: #4a6080 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.8px !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.4rem !important; font-family: 'JetBrains Mono', monospace !important; }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #030712; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#00f5ff,#7c3aed); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown("""
<div class="page-header">
    <div class="page-title">📊 Stock Explorer</div>
    <div class="page-subtitle">Analyse historical prices, 3D volume depth, technical indicators and performance statistics.</div>
</div>
""", unsafe_allow_html=True)

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
    show_volume      = st.checkbox("Volume bars", value=True)
    show_sma         = st.checkbox("Moving averages (SMA 20, 50)", value=True)
    show_bollinger   = st.checkbox("Bollinger Bands", value=False)
    show_rsi         = st.checkbox("RSI (14)", value=True)
    show_3d_volume   = st.checkbox("3D Volume Surface", value=False)

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

df         = load_data(ticker, start_date)
indicators = load_indicators(ticker, start_date)
company_name = COMPANY_INFO.get(ticker, {}).get("name", ticker)

if df.empty:
    st.warning(f"No data found for **{ticker}**. Please run the ingestion pipeline first.")
    st.code("python -m src.ingestion.fetch_market_data", language="bash")
    st.stop()

# ── KPI Row ────────────────────────────────────────────────────────────────
latest        = df.iloc[-1]
first         = df.iloc[0]
period_return = (latest["close_price"] - first["close_price"]) / first["close_price"] * 100
daily_return  = (df["close_price"].pct_change() * 100).iloc[-1]
high_52w      = df["high_price"].max()
low_52w       = df["low_price"].min()
avg_vol       = df["volume"].mean()

st.markdown(f"""
<div class="sec-label" style="margin-bottom:0.8rem">
    {company_name} ({ticker})
    <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Close Price",   f"₹{latest['close_price']:,.2f}", f"{daily_return:+.2f}%")
c2.metric("Period Return", f"{period_return:+.2f}%")
c3.metric("52W High",      f"₹{high_52w:,.2f}")
c4.metric("52W Low",       f"₹{low_52w:,.2f}")
c5.metric("Avg Volume",    f"{avg_vol/1e6:.1f}M")

st.markdown('<div style="margin-bottom:1.5rem"></div>', unsafe_allow_html=True)

# ── 3D Volume Surface (optional) ───────────────────────────────────────────
if show_3d_volume and not df.empty:
    st.markdown("""
    <div class="sec-label">
        📊 3D Volume Surface — Price × Volume × Time
        <div class="sec-line"></div>
    </div>
    """, unsafe_allow_html=True)

    prices  = df["close_price"].values
    volumes = df["volume"].values
    n       = len(prices)
    x_idx   = np.arange(n)

    depth_bands = 6
    depth       = np.linspace(0, 1, depth_bands)
    Z_price     = np.outer(prices, np.ones(depth_bands))
    Z_vol       = np.outer(volumes / volumes.max(), np.ones(depth_bands))

    fig3d_vol = go.Figure()
    fig3d_vol.add_trace(go.Surface(
        x=np.outer(x_idx, np.ones(depth_bands)),
        y=np.outer(np.ones(n), depth * volumes.max() * 0.3),
        z=Z_price,
        surfacecolor=Z_vol,
        colorscale=[
            [0.0, "rgba(0,20,80,0.9)"],
            [0.4, "rgba(0,100,220,0.9)"],
            [0.7, "rgba(0,245,255,0.9)"],
            [1.0, "rgba(0,255,136,0.9)"],
        ],
        showscale=True,
        colorbar=dict(title="Volume", tickfont=dict(color="#4a6080"), titlefont=dict(color="#4a6080")),
        hovertemplate="Price: ₹%{z:,.0f}<extra></extra>",
    ))

    fig3d_vol.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title="Time", titlefont=dict(color="#4a6080")),
            yaxis=dict(showgrid=False, showticklabels=False, title="Volume Depth", titlefont=dict(color="#4a6080")),
            zaxis=dict(showgrid=True, gridcolor="rgba(0,245,255,0.06)", color="#4a6080", title="Price ₹", tickprefix="₹", tickformat=",.0f", titlefont=dict(color="#00f5ff")),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig3d_vol, use_container_width=True)

# ── Price chart ────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-label">
    📈 Price Chart
    <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

rows        = 2 if (show_volume or show_rsi) else 1
row_heights = [0.65, 0.35] if rows == 2 else [1.0]

fig = make_subplots(
    rows=rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=row_heights,
)

# Price trace
if show_candlestick:
    fig.add_trace(go.Candlestick(
        x=df["trade_date"],
        open=df["open_price"], high=df["high_price"],
        low=df["low_price"],   close=df["close_price"],
        name="OHLC",
        increasing_line_color="#00ff88", increasing_fillcolor="rgba(0,255,136,0.7)",
        decreasing_line_color="#ff4d6a", decreasing_fillcolor="rgba(255,77,106,0.7)",
    ), row=1, col=1)
else:
    fig.add_trace(go.Scatter(
        x=df["trade_date"], y=df["close_price"],
        mode="lines", name="Close",
        line=dict(color="#00f5ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,245,255,0.04)",
    ), row=1, col=1)

# Moving averages
if show_sma and not indicators.empty:
    for col_name, color, label in [
        ("sma_20", "#f59e0b", "SMA 20"),
        ("sma_50", "#a78bfa", "SMA 50"),
    ]:
        if col_name in indicators.columns:
            fig.add_trace(go.Scatter(
                x=indicators["trade_date"], y=indicators[col_name],
                mode="lines", name=label,
                line=dict(color=color, width=1.5, dash="dot"),
            ), row=1, col=1)

# Bollinger Bands
if show_bollinger and not indicators.empty:
    if "bollinger_upper" in indicators.columns:
        fig.add_trace(go.Scatter(
            x=indicators["trade_date"], y=indicators["bollinger_upper"],
            name="BB Upper", line=dict(color="rgba(0,245,255,0.3)", width=1),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=indicators["trade_date"], y=indicators["bollinger_lower"],
            name="BB Lower", line=dict(color="rgba(0,245,255,0.3)", width=1),
            fill="tonexty", fillcolor="rgba(0,245,255,0.03)",
        ), row=1, col=1)

# Volume
if show_volume:
    vol_colors = ["rgba(0,255,136,0.5)" if r >= 0 else "rgba(255,77,106,0.5)"
                  for r in df["close_price"].pct_change().fillna(0)]
    fig.add_trace(go.Bar(
        x=df["trade_date"], y=df["volume"],
        name="Volume", marker_color=vol_colors,
        marker_line_width=0,
    ), row=2, col=1)

# RSI
if show_rsi and not indicators.empty and "rsi_14" in indicators.columns:
    rsi_row = 2
    fig.add_trace(go.Scatter(
        x=indicators["trade_date"], y=indicators["rsi_14"],
        name="RSI 14", line=dict(color="#f59e0b", width=1.5),
    ), row=rsi_row, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,77,106,0.5)", row=rsi_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.5)", row=rsi_row, col=1)

fig.update_layout(
    template="plotly_dark",
    height=560,
    xaxis_rangeslider_visible=False,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=11, color="#4a6080"), bgcolor="rgba(0,0,0,0)",
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(3,7,18,0.6)",
    xaxis=dict(showgrid=False, color="#2d4560", tickformat="%b '%y"),
    yaxis=dict(
        showgrid=True, gridcolor="rgba(0,245,255,0.04)",
        color="#4a6080", tickprefix="₹", tickformat=",.0f",
    ),
)
st.plotly_chart(fig, use_container_width=True)

# ── Statistics ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-label">
    📐 Summary Statistics
    <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

daily_ret = df["close_price"].pct_change().dropna()
stats = {
    "Trading Days":          len(df),
    "Latest Close (₹)":     f"₹{latest['close_price']:,.2f}",
    "Period Return":         f"{period_return:+.2f}%",
    "Daily Return (mean)":   f"{daily_ret.mean()*100:.4f}%",
    "Annualized Return":     f"{daily_ret.mean()*252*100:.2f}%",
    "Annualized Volatility": f"{daily_ret.std()*252**0.5*100:.2f}%",
    "Max Daily Gain":        f"{daily_ret.max()*100:.2f}%",
    "Max Daily Loss":        f"{daily_ret.min()*100:.2f}%",
    "Positive Days":         f"{(daily_ret>0).mean()*100:.1f}%",
    "52W High":              f"₹{high_52w:,.2f}",
    "52W Low":               f"₹{low_52w:,.2f}",
    "Avg Volume":            f"{avg_vol/1e6:.2f}M",
}

stats_html = "".join([
    f'<div class="stat-row"><span class="stat-metric">{k}</span><span class="stat-value">{v}</span></div>'
    for k, v in stats.items()
])
st.markdown(f'<div class="stats-card">{stats_html}</div>', unsafe_allow_html=True)
