"""
StockVision AI — Streamlit Dashboard (Premium 3D Edition)
==========================================================
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
import numpy as np
from datetime import datetime, timedelta

from src.utils.config import (
    COMPANY_INFO, BENCHMARK_TICKER, BENCHMARK_NAME,
    ALL_TICKERS, STOCK_UNIVERSE,
)
from src.database.queries import (
    get_stock_prices,
    get_all_tickers,
    get_latest_predictions,
    get_model_metrics,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockVision AI — Market Analytics & Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/your-repo/stockvision-ai",
        "About": "StockVision AI — Market Analytics & Forecasting Platform",
    },
)

# ── Premium 3D CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #030712;
        color: #e2e8f0;
    }

    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1440px;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #030712; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#00f5ff,#7c3aed); border-radius: 10px; }

    /* ── Animated Hero ── */
    .hero-wrap {
        position: relative;
        border-radius: 28px;
        overflow: hidden;
        margin-bottom: 2rem;
        padding: 3.5rem 4rem 3rem;
        background: radial-gradient(ellipse at 70% 20%, rgba(124,58,237,0.18) 0%, transparent 55%),
                    radial-gradient(ellipse at 10% 80%, rgba(0,245,255,0.12) 0%, transparent 50%),
                    linear-gradient(135deg, #050b18 0%, #0a1020 60%, #060e1c 100%);
        border: 1px solid rgba(0,245,255,0.12);
    }

    /* Animated grid overlay */
    .hero-wrap::before {
        content: '';
        position: absolute; inset: 0;
        background-image:
            linear-gradient(rgba(0,245,255,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,245,255,0.035) 1px, transparent 1px);
        background-size: 48px 48px;
        animation: gridMove 8s linear infinite;
        pointer-events: none;
    }

    @keyframes gridMove {
        0%   { background-position: 0 0; }
        100% { background-position: 48px 48px; }
    }

    /* Floating neon orbs */
    .hero-wrap::after {
        content: '';
        position: absolute;
        width: 360px; height: 360px;
        top: -100px; right: -80px;
        background: radial-gradient(circle, rgba(124,58,237,0.25) 0%, transparent 65%);
        border-radius: 50%;
        animation: orbFloat 6s ease-in-out infinite alternate;
        pointer-events: none;
    }

    @keyframes orbFloat {
        0%   { transform: translate(0,0) scale(1); }
        100% { transform: translate(20px,-20px) scale(1.1); }
    }

    .hero-badge {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(0,245,255,0.08);
        border: 1px solid rgba(0,245,255,0.25);
        border-radius: 100px; padding: 5px 16px;
        font-size: 0.72rem; font-weight: 700; color: #00f5ff;
        letter-spacing: 1.5px; text-transform: uppercase;
        margin-bottom: 1.4rem;
        box-shadow: 0 0 16px rgba(0,245,255,0.12);
        position: relative; z-index: 1;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3.4rem; font-weight: 800;
        color: #f8fafc; line-height: 1.1;
        margin: 0 0 1rem 0; letter-spacing: -1.5px;
        position: relative; z-index: 1;
    }

    .hero-title .glow-text {
        background: linear-gradient(90deg, #00f5ff 0%, #7c3aed 50%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 0 20px rgba(0,245,255,0.4));
        animation: shimmer 3s ease-in-out infinite;
        background-size: 200% auto;
    }

    @keyframes shimmer {
        0%   { background-position: 0% center; }
        50%  { background-position: 100% center; }
        100% { background-position: 0% center; }
    }

    .hero-subtitle {
        font-size: 1.05rem; color: #7d8fa8;
        max-width: 600px; line-height: 1.75;
        margin: 0 0 1.5rem 0;
        position: relative; z-index: 1;
    }

    .live-pill {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(0,255,136,0.07);
        border: 1px solid rgba(0,255,136,0.2);
        border-radius: 100px; padding: 5px 14px;
        font-size: 0.75rem; color: #00ff88; font-weight: 600;
        position: relative; z-index: 1;
        box-shadow: 0 0 12px rgba(0,255,136,0.1);
    }

    .pulse-ring {
        width: 8px; height: 8px;
        background: #00ff88; border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(0,255,136,0.5);
        animation: ringPulse 1.8s ease-in-out infinite;
    }

    @keyframes ringPulse {
        0%   { box-shadow: 0 0 0 0   rgba(0,255,136,0.5); }
        70%  { box-shadow: 0 0 0 8px rgba(0,255,136,0); }
        100% { box-shadow: 0 0 0 0   rgba(0,255,136,0); }
    }

    .hero-stats {
        display: flex; gap: 3rem; flex-wrap: wrap;
        margin-top: 2rem; padding-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        position: relative; z-index: 1;
    }

    .h-stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem; font-weight: 700; color: #f8fafc;
        display: block; line-height: 1;
    }

    .h-stat-label {
        font-size: 0.68rem; color: #4a6080;
        text-transform: uppercase; letter-spacing: 1px; font-weight: 600;
        margin-top: 4px; display: block;
    }

    /* ── Glass KPI cards ── */
    .kpi-glass {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(255,255,255,0.07);
        position: relative; overflow: hidden;
        transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
        margin-bottom: 0;
    }

    .kpi-glass::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0;
        height: 1px;
        border-radius: 20px 20px 0 0;
    }

    .kpi-glass.cyan::before   { background: linear-gradient(90deg,#00f5ff,transparent); }
    .kpi-glass.violet::before { background: linear-gradient(90deg,#7c3aed,transparent); }
    .kpi-glass.green::before  { background: linear-gradient(90deg,#00ff88,transparent); }
    .kpi-glass.orange::before { background: linear-gradient(90deg,#f59e0b,transparent); }

    .kpi-glass:hover {
        background: rgba(255,255,255,0.055);
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 20px 60px rgba(0,0,0,0.5),
                    0 0 40px rgba(0,245,255,0.06);
        border-color: rgba(0,245,255,0.18);
    }

    .kpi-glow-icon {
        width: 52px; height: 52px; border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem; margin-bottom: 1rem;
    }

    .kpi-glow-icon.cyan   { background: rgba(0,245,255,0.1);  box-shadow: 0 0 20px rgba(0,245,255,0.15); }
    .kpi-glow-icon.violet { background: rgba(124,58,237,0.12); box-shadow: 0 0 20px rgba(124,58,237,0.2); }
    .kpi-glow-icon.green  { background: rgba(0,255,136,0.1);   box-shadow: 0 0 20px rgba(0,255,136,0.15); }
    .kpi-glow-icon.orange { background: rgba(245,158,11,0.1);  box-shadow: 0 0 20px rgba(245,158,11,0.15); }

    .kpi-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem; font-weight: 700; color: #f8fafc;
        line-height: 1; margin-bottom: 0.3rem; display: block;
    }

    .kpi-label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #4a6080; }
    .kpi-sub   { font-size: 0.72rem; color: #2d4560; margin-top: 4px; }

    /* ── Section label ── */
    .sec-label {
        font-family: 'Syne', sans-serif;
        font-size: 1.15rem; font-weight: 700; color: #e2e8f0;
        margin: 0 0 0.4rem 0; display: flex; align-items: center; gap: 10px;
    }

    .sec-line {
        height: 1px; flex: 1;
        background: linear-gradient(90deg, rgba(0,245,255,0.3), transparent);
    }

    /* ── Ticker cards ── */
    .tick-card {
        background: rgba(255,255,255,0.025);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 1.1rem 1.3rem;
        transition: all 0.3s ease; height: 100%;
        position: relative; overflow: hidden;
    }

    .tick-card::after {
        content: '';
        position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
        background: transparent;
        transition: background 0.3s ease;
    }

    .tick-card:hover {
        border-color: rgba(0,245,255,0.22);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 30px rgba(0,245,255,0.06);
    }

    .tick-card:hover::after {
        background: linear-gradient(90deg, #00f5ff, #7c3aed);
    }

    .tick-sym   { font-size: 0.68rem; font-weight: 800; color: #00f5ff; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 2px; }
    .tick-name  { font-size: 0.7rem; color: #3d5268; margin-bottom: 0.6rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .tick-price { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #f8fafc; margin-bottom: 5px; }

    .chg-up   { font-size: 0.75rem; font-weight: 700; padding: 3px 9px; border-radius: 8px; display: inline-block; color: #00ff88; background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.2); }
    .chg-down { font-size: 0.75rem; font-weight: 700; padding: 3px 9px; border-radius: 8px; display: inline-block; color: #ff4d6a; background: rgba(255,77,106,0.1); border: 1px solid rgba(255,77,106,0.2); }
    .chg-flat { font-size: 0.75rem; font-weight: 700; padding: 3px 9px; border-radius: 8px; display: inline-block; color: #7d8fa8; background: rgba(125,143,168,0.08); border: 1px solid rgba(125,143,168,0.15); }

    /* ── Feature cards ── */
    .feat-card {
        background: rgba(255,255,255,0.025);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px; padding: 1.8rem;
        height: 100%; transition: all 0.35s ease; position: relative; overflow: hidden;
    }

    .feat-card:hover {
        border-color: rgba(0,245,255,0.2);
        box-shadow: 0 24px 64px rgba(0,0,0,0.6), 0 0 40px rgba(0,245,255,0.05);
        transform: translateY(-6px);
    }

    .feat-icon {
        width: 52px; height: 52px; border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem; margin-bottom: 1.2rem;
    }

    .fi-c { background: rgba(0,245,255,0.1);  box-shadow: 0 0 20px rgba(0,245,255,0.12); }
    .fi-g { background: rgba(0,255,136,0.1);  box-shadow: 0 0 20px rgba(0,255,136,0.12); }
    .fi-v { background: rgba(124,58,237,0.12); box-shadow: 0 0 20px rgba(124,58,237,0.15); }
    .fi-o { background: rgba(245,158,11,0.1);  box-shadow: 0 0 20px rgba(245,158,11,0.12); }

    .feat-title { font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem; }
    .feat-desc  { font-size: 0.83rem; color: #4a6080; line-height: 1.65; }
    .feat-tag   { display: inline-block; font-size: 0.67rem; font-weight: 700; padding: 3px 10px; border-radius: 100px; margin-top: 1rem; text-transform: uppercase; letter-spacing: 0.8px; }

    .ft-c { color: #00f5ff; background: rgba(0,245,255,0.1); border: 1px solid rgba(0,245,255,0.2); }
    .ft-g { color: #00ff88; background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.2); }
    .ft-v { color: #a78bfa; background: rgba(124,58,237,0.1); border: 1px solid rgba(124,58,237,0.2); }
    .ft-o { color: #fbbf24; background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.2); }

    /* ── Disclaimer ── */
    .disc-bar {
        background: rgba(245,158,11,0.05);
        border: 1px solid rgba(245,158,11,0.15);
        border-left: 3px solid #f59e0b;
        border-radius: 12px; padding: 1rem 1.4rem;
        font-size: 0.8rem; color: #92702a;
        margin-top: 2rem; line-height: 1.7;
    }

    /* ── Divider ── */
    .h-div { height: 1px; background: rgba(255,255,255,0.04); margin: 2rem 0; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #030712 0%, #060e1c 100%) !important;
        border-right: 1px solid rgba(0,245,255,0.08) !important;
    }

    /* Style Streamlit's native page navigation links */
    [data-testid="stSidebarNav"] {
        padding-top: 0.5rem;
    }

    [data-testid="stSidebarNav"] ul {
        padding-left: 0 !important;
        list-style: none !important;
    }

    [data-testid="stSidebarNav"] a {
        display: flex !important;
        align-items: center !important;
        padding: 9px 12px !important;
        border-radius: 10px !important;
        font-size: 0.84rem !important;
        font-family: 'Inter', sans-serif !important;
        color: #4a6080 !important;
        text-decoration: none !important;
        transition: all 0.22s ease !important;
        border: 1px solid transparent !important;
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
        color: #00f5ff !important;
        font-weight: 600 !important;
        box-shadow: 0 0 18px rgba(0,245,255,0.07) !important;
    }

    [data-testid="stSidebarNav"] span {
        color: inherit !important;
        font-weight: inherit !important;
    }

    /* Nav section label */
    .nav-sec-label {
        font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1.5px; color: #1e3048;
        margin: 0.6rem 0 0.4rem 0.2rem;
    }

    .sb-logo {
        font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800;
        background: linear-gradient(90deg, #00f5ff, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; display: block; margin-bottom: 2px;
        filter: drop-shadow(0 0 12px rgba(0,245,255,0.3));
    }

    .sb-tagline { font-size: 0.65rem; color: #2d4560; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; }

    .sb-info {
        background: rgba(0,245,255,0.03);
        border: 1px solid rgba(0,245,255,0.08);
        border-radius: 12px; padding: 0.9rem 1rem;
        margin-top: 0.5rem;
    }

    .sb-row { display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.76rem; color: #2d4560 !important; }
    .sb-val { color: #8ba4c0 !important; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; }

    /* ── Metric override ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.025) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
        padding: 1rem 1.2rem !important;
        backdrop-filter: blur(8px) !important;
    }
    [data-testid="stMetricLabel"] { color: #4a6080 !important; font-size: 0.73rem !important; text-transform: uppercase; letter-spacing: 0.8px !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.6rem !important; font-family: 'JetBrains Mono', monospace !important; }
    [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_prices(ticker: str, days: int = 365) -> pd.DataFrame:
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


def format_change(value: float):
    if value is None or pd.isna(value):
        return "N/A", "chg-flat"
    if value > 0:
        return f"▲ {abs(value):.2f}%", "chg-up"
    if value < 0:
        return f"▼ {abs(value):.2f}%", "chg-down"
    return "0.00%", "chg-flat"


# ── Home Page ──────────────────────────────────────────────────────────────

def render_home():
    now_str = datetime.now().strftime("%d %b %Y · %I:%M %p IST")

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-badge">⚡ AI-Powered &nbsp;·&nbsp; NSE &nbsp;·&nbsp; NIFTY 50</div>
        <h1 class="hero-title">Stock<span class="glow-text">Vision AI</span></h1>
        <p class="hero-subtitle">
            Institutional-grade market analytics, risk intelligence and ML-driven
            price forecasting for the NIFTY 50 universe — built for data-driven investors.
        </p>
        <div class="live-pill">
            <div class="pulse-ring"></div>
            Live Data &nbsp;·&nbsp; {now_str}
        </div>
        <div class="hero-stats">
            <div>
                <span class="h-stat-val">{len(ALL_TICKERS)}</span>
                <span class="h-stat-label">Stocks Tracked</span>
            </div>
            <div>
                <span class="h-stat-val">{len(STOCK_UNIVERSE)}</span>
                <span class="h-stat-label">Sectors</span>
            </div>
            <div>
                <span class="h-stat-val">50+</span>
                <span class="h-stat-label">Features Engineered</span>
            </div>
            <div>
                <span class="h-stat-val">5+ yr</span>
                <span class="h-stat-label">Historical Data</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI glass cards ───────────────────────────────────────────────────
    kpi_data = [
        ("cyan",   "💹", str(len(ALL_TICKERS)), "Stocks Monitored",   "NIFTY 50 Universe"),
        ("green",  "🏭", str(len(STOCK_UNIVERSE)), "Sectors Covered", "IT · Banking · Energy · Auto"),
        ("violet", "🤖", "3",                   "ML Models Deployed",  "XGBoost · RF · Ridge"),
        ("orange", "📅", "5+ yrs",              "Historical Data",     "Daily OHLCV via yfinance"),
    ]

    kpi_cols = st.columns(4)
    for col, (color, icon, value, label, sub) in zip(kpi_cols, kpi_data):
        col.markdown(f"""
        <div class="kpi-glass {color}">
            <div class="kpi-glow-icon {color}">{icon}</div>
            <span class="kpi-val">{value}</span>
            <div class="kpi-label">{label}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom:2.5rem"></div>', unsafe_allow_html=True)

    # ── Market Snapshot ───────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-label">
        📡 Market Snapshot
        <div class="sec-line"></div>
    </div>
    """, unsafe_allow_html=True)

    snap_cols = st.columns(len(ALL_TICKERS[:6]))
    for col, ticker in zip(snap_cols, ALL_TICKERS[:6]):
        df = load_prices(ticker, days=5)
        short = ticker.replace(".NS", "")
        cname = COMPANY_INFO.get(ticker, {}).get("name", ticker)

        if df.empty or len(df) < 2:
            col.markdown(f"""
            <div class="tick-card">
                <div class="tick-sym">{short}</div>
                <div class="tick-name">{cname[:20]}</div>
                <div class="tick-price">—</div>
                <span class="chg-flat">N/A</span>
            </div>""", unsafe_allow_html=True)
            continue

        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        chg    = (latest["close_price"] - prev["close_price"]) / prev["close_price"] * 100
        chg_str, chg_cls = format_change(chg)

        col.markdown(f"""
        <div class="tick-card">
            <div class="tick-sym">{short}</div>
            <div class="tick-name">{cname[:22]}</div>
            <div class="tick-price">₹{latest['close_price']:,.0f}</div>
            <span class="{chg_cls}">{chg_str}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom:2.5rem"></div>', unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────
    chart_col, side_col = st.columns([3, 2], gap="large")

    with chart_col:
        st.markdown("""
        <div class="sec-label">
            📈 NIFTY 50 — 3D Surface Overview
            <div class="sec-line"></div>
        </div>
        """, unsafe_allow_html=True)

        nifty_df = load_prices(BENCHMARK_TICKER, days=365)
        if not nifty_df.empty:
            first_val  = nifty_df["close_price"].iloc[0]
            last_val   = nifty_df["close_price"].iloc[-1]
            ytd_return = (last_val - first_val) / first_val * 100
            max_val    = nifty_df["close_price"].max()

            m1, m2, m3 = st.columns(3)
            m1.metric("Current",       f"₹{last_val:,.0f}")
            m2.metric("1-Year Return", f"{ytd_return:+.2f}%")
            m3.metric("52-Week High",  f"₹{max_val:,.0f}")

            # 3D Surface Chart — price over time with depth wave
            prices   = nifty_df["close_price"].values
            n        = len(prices)
            x_idx    = np.arange(n)
            depth    = np.linspace(0, 1, 8)
            Z        = np.outer(prices, np.ones(len(depth)))

            # Add subtle wave distortion on depth axis
            for di, d in enumerate(depth):
                wave = np.sin(np.linspace(0, 2*np.pi, n) + d*np.pi) * prices.std() * 0.06 * (1 - d)
                Z[:, di] = prices + wave

            fig = go.Figure(data=[go.Surface(
                x=np.outer(x_idx, np.ones(len(depth))),
                y=np.outer(np.ones(n), depth),
                z=Z,
                colorscale=[
                    [0.0,  "rgba(0,20,60,1)"],
                    [0.3,  "rgba(0,80,180,1)"],
                    [0.6,  "rgba(0,200,255,1)"],
                    [0.85, "rgba(120,0,255,1)"],
                    [1.0,  "rgba(0,255,136,1)"],
                ],
                opacity=0.92,
                showscale=False,
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="#00f5ff", project_z=False),
                ),
                hovertemplate="Price: ₹%{z:,.0f}<extra></extra>",
            )])

            fig.update_layout(
                scene=dict(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", backgroundcolor="rgba(0,0,0,0)"),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", backgroundcolor="rgba(0,0,0,0)"),
                    zaxis=dict(
                        showgrid=True, gridcolor="rgba(0,245,255,0.06)",
                        zeroline=False, color="#4a6080",
                        title="Price ₹", titlefont=dict(color="#00f5ff", size=10),
                        tickprefix="₹", tickformat=",.0f",
                    ),
                    bgcolor="rgba(0,0,0,0)",
                    camera=dict(eye=dict(x=1.8, y=-1.4, z=0.85)),
                ),
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ Run the ingestion pipeline to load market data.")

    with side_col:
        st.markdown("""
        <div class="sec-label">
            🏭 Sector Distribution — 3D
            <div class="sec-line"></div>
        </div>
        """, unsafe_allow_html=True)

        sectors = {}
        for ticker, info in COMPANY_INFO.items():
            sec = info.get("sector", "Other")
            if sec != "Benchmark":
                sectors[sec] = sectors.get(sec, 0) + 1

        sec_labels = list(sectors.keys())
        sec_values = list(sectors.values())
        neon_colors = ["#00f5ff", "#00ff88", "#7c3aed", "#f59e0b"]

        fig2 = go.Figure()
        for i, (sec, val) in enumerate(zip(sec_labels, sec_values)):
            color = neon_colors[i % len(neon_colors)]
            fig2.add_trace(go.Bar(
                x=[val], y=[sec], orientation="h",
                marker=dict(
                    color=color,
                    opacity=0.85,
                    line=dict(color=color, width=0),
                ),
                showlegend=False,
                hovertemplate=f"<b>{sec}</b><br>{val} stocks<extra></extra>",
            ))
            # Glow effect via duplicate transparent bar
            fig2.add_trace(go.Bar(
                x=[val], y=[sec], orientation="h",
                marker=dict(color=color, opacity=0.15, line=dict(color=color, width=8)),
                showlegend=False, hoverinfo="skip",
            ))

        fig2.update_layout(
            template="plotly_dark", height=210,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="overlay",
            xaxis=dict(showgrid=False, color="#2d4560", dtick=1),
            yaxis=dict(showgrid=False, color="#7d8fa8"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <div class="sec-label" style="margin-top:1.2rem">
            🔮 Latest AI Forecasts
            <div class="sec-line"></div>
        </div>
        """, unsafe_allow_html=True)

        preds_df = load_predictions()
        if not preds_df.empty:
            display_cols = ["ticker", "model_name", "predicted_direction", "prediction_probability"]
            existing = [c for c in display_cols if c in preds_df.columns]
            st.dataframe(
                preds_df[existing].head(8).rename(columns={
                    "ticker": "Ticker",
                    "model_name": "Model",
                    "predicted_direction": "Signal",
                    "prediction_probability": "Confidence",
                }),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0, max_value=1, format="%.2f"
                    )
                },
            )
        else:
            st.info("🤖 Run training & prediction pipeline to see forecasts.")

    # ── Feature nav cards ─────────────────────────────────────────────────
    st.markdown('<div class="h-div"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-label">
        🚀 Explore the Platform
        <div class="sec-line"></div>
    </div>
    """, unsafe_allow_html=True)

    features = [
        ("fi-c", "ft-c", "📊", "Stock Explorer",
         "Interactive 3D candlestick charts, RSI, MACD, Bollinger Bands and full technical analysis suite.",
         "Charts & Indicators"),
        ("fi-g", "ft-g", "🔮", "Forecast Centre",
         "ML-powered next-day and 5-day price direction predictions with animated confidence gauges.",
         "AI Predictions"),
        ("fi-v", "ft-v", "🧪", "Model Laboratory",
         "3D model comparison scatter, leaderboard rankings, feature importance and evaluation metrics.",
         "ML Insights"),
        ("fi-o", "ft-o", "🔍", "Data Quality",
         "Monitor pipeline health, data freshness, missing records and ingestion integrity.",
         "Pipeline Health"),
    ]

    fc = st.columns(4, gap="medium")
    for col, (fi_cls, tag_cls, icon, title, desc, tag) in zip(fc, features):
        col.markdown(f"""
        <div class="feat-card">
            <div class="feat-icon {fi_cls}">{icon}</div>
            <div class="feat-title">{title}</div>
            <div class="feat-desc">{desc}</div>
            <span class="feat-tag {tag_cls}">{tag}</span>
        </div>""", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disc-bar">
        ⚠️ <strong>Important Disclaimer:</strong> All forecasts and analytics are generated
        from historical market data for <strong>educational and research purposes only</strong>.
        They do not constitute financial or investment advice. Past market patterns do not
        guarantee future performance. Always consult a qualified financial advisor before
        making investment decisions. StockVision AI is a portfolio project.
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Brand logo ────────────────────────────────────────────────────────
    st.markdown("""
    <span class="sb-logo">📈 StockVision</span>
    <span class="sb-tagline">AI Market Analytics Platform</span>
    """, unsafe_allow_html=True)

    st.markdown('<div class="h-div" style="margin:0.6rem 0 0 0;"></div>', unsafe_allow_html=True)

    # ── Navigation label (Streamlit renders the actual links natively above this) ──
    st.markdown('<div class="nav-sec-label">Navigation</div>', unsafe_allow_html=True)

    # Streamlit's built-in multi-page nav links are rendered automatically here
    # The CSS above styles them to match the neon theme

    st.markdown('<div class="h-div" style="margin:0.6rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-sec-label">Platform Info</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sb-info">
        <div class="sb-row"><span>Universe</span><span class="sb-val">NIFTY 50</span></div>
        <div class="sb-row"><span>Stocks</span><span class="sb-val">{len(ALL_TICKERS)} tracked</span></div>
        <div class="sb-row"><span>Forecast</span><span class="sb-val">1d &amp; 5d</span></div>
        <div class="sb-row"><span>Models</span><span class="sb-val">XGB · RF · LR</span></div>
        <div class="sb-row" style="margin-bottom:0"><span>Data</span><span class="sb-val">yfinance</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="h-div" style="margin:0.8rem 0;"></div>', unsafe_allow_html=True)
    st.caption("Built for Data Analyst Portfolio · 2024\nPowered by XGBoost · Streamlit · Plotly")


# ── Render ─────────────────────────────────────────────────────────────────
render_home()
