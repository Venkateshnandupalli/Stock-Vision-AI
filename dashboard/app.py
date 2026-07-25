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

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #060d18;
        color: #e2e8f0;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px;
    }

    /* ── Hide Streamlit default chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Custom Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1b2a; }
    ::-webkit-scrollbar-thumb { background: #2a4a6b; border-radius: 10px; }

    /* ── Hero ── */
    .hero-container {
        background: linear-gradient(135deg, #0a1628 0%, #0d2137 40%, #0a1f35 70%, #071220 100%);
        border: 1px solid rgba(56, 139, 253, 0.15);
        border-radius: 24px;
        padding: 3rem 3.5rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -60%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(56,139,253,0.08) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -5%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0,210,180,0.05) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56,139,253,0.12);
        border: 1px solid rgba(56,139,253,0.3);
        border-radius: 100px;
        padding: 4px 14px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #58a6ff;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #f0f6ff;
        line-height: 1.15;
        margin: 0 0 0.8rem 0;
        letter-spacing: -1px;
    }

    .hero-title span {
        background: linear-gradient(135deg, #388bfd 0%, #00d2b4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #8b9fbe;
        max-width: 620px;
        line-height: 1.7;
        margin: 0 0 1.2rem 0;
    }

    .hero-stats {
        display: flex;
        gap: 2.5rem;
        flex-wrap: wrap;
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.07);
    }

    .hero-stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        font-family: 'Space Grotesk', sans-serif;
        display: block;
    }

    .hero-stat-label {
        font-size: 0.75rem;
        color: #5c7a9e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
    }

    /* ── Pulse dot ── */
    .live-row {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.78rem;
        color: #5c7a9e;
        margin-bottom: 0;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #3fb950;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s ease-in-out infinite;
        box-shadow: 0 0 0 0 rgba(63,185,80,0.4);
    }

    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(63,185,80,0.4); }
        70%  { box-shadow: 0 0 0 8px rgba(63,185,80,0); }
        100% { box-shadow: 0 0 0 0 rgba(63,185,80,0); }
    }

    .live-text { color: #3fb950; font-weight: 600; }

    /* ── Section headers ── */
    .section-header-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #c9d1d9;
        margin: 0 0 0.5rem 0;
    }

    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, rgba(56,139,253,0.5), transparent);
        border-radius: 2px;
        margin-bottom: 1.2rem;
    }

    /* ── KPI cards ── */
    .kpi-card {
        background: linear-gradient(145deg, #0d1b2a, #111f30);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        margin-bottom: 0;
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        border-radius: 16px 16px 0 0;
    }

    .kpi-card.blue::before   { background: linear-gradient(90deg,#388bfd,#58a6ff); }
    .kpi-card.green::before  { background: linear-gradient(90deg,#3fb950,#56d364); }
    .kpi-card.teal::before   { background: linear-gradient(90deg,#00d2b4,#00b4d8); }
    .kpi-card.purple::before { background: linear-gradient(90deg,#bc8cff,#d29af5); }

    .kpi-card:hover {
        border-color: rgba(56,139,253,0.25);
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.3);
    }

    .kpi-icon  { font-size: 1.6rem; margin-bottom: 0.6rem; display: block; }
    .kpi-value { font-family: 'Space Grotesk',sans-serif; font-size: 2rem; font-weight: 700; color: #e2e8f0; line-height: 1; margin-bottom: 0.3rem; }
    .kpi-label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: #5c7a9e; }
    .kpi-sub   { font-size: 0.73rem; color: #3c5775; margin-top: 3px; }

    /* ── Ticker cards ── */
    .ticker-card {
        background: linear-gradient(145deg,#0d1b2a,#0f2238);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        transition: all 0.25s ease;
        height: 100%;
    }

    .ticker-card:hover {
        border-color: rgba(56,139,253,0.3);
        box-shadow: 0 8px 24px rgba(56,139,253,0.1);
        transform: translateY(-2px);
    }

    .ticker-symbol { font-size: 0.72rem; font-weight: 700; color: #388bfd; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
    .ticker-name   { font-size: 0.72rem; color: #5c7a9e; margin-bottom: 0.5rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .ticker-price  { font-family: 'Space Grotesk',sans-serif; font-size: 1.25rem; font-weight: 700; color: #e2e8f0; margin-bottom: 4px; }

    .ticker-change       { font-size: 0.78rem; font-weight: 600; padding: 2px 8px; border-radius: 6px; display: inline-block; }
    .change-up   { color: #3fb950; background: rgba(63,185,80,0.1); }
    .change-down { color: #f85149; background: rgba(248,81,73,0.1); }
    .change-flat { color: #8b9fbe; background: rgba(139,159,190,0.1); }

    /* ── Feature cards ── */
    .feature-card {
        background: linear-gradient(145deg,#0d1b2a,#0f2238);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        border-color: rgba(56,139,253,0.25);
        box-shadow: 0 16px 48px rgba(0,0,0,0.3);
        transform: translateY(-4px);
    }

    .feature-icon  { width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; margin-bottom: 1rem; }
    .fi-blue       { background: rgba(56,139,253,0.15); }
    .fi-green      { background: rgba(63,185,80,0.15); }
    .fi-teal       { background: rgba(0,210,180,0.15); }
    .fi-purple     { background: rgba(188,140,255,0.15); }

    .feature-title { font-family: 'Space Grotesk',sans-serif; font-size: 1rem; font-weight: 600; color: #c9d1d9; margin-bottom: 0.4rem; }
    .feature-desc  { font-size: 0.83rem; color: #5c7a9e; line-height: 1.6; }

    .feature-tag  { display: inline-block; font-size: 0.68rem; font-weight: 600; padding: 2px 8px; border-radius: 100px; margin-top: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .tag-blue     { background: rgba(56,139,253,0.15); color: #58a6ff; }
    .tag-green    { background: rgba(63,185,80,0.15); color: #3fb950; }
    .tag-teal     { background: rgba(0,210,180,0.15); color: #00d2b4; }
    .tag-purple   { background: rgba(188,140,255,0.15); color: #bc8cff; }

    /* ── Horizontal rule ── */
    .h-divider { height: 1px; background: rgba(255,255,255,0.05); margin: 2rem 0; }

    /* ── Disclaimer ── */
    .disclaimer-bar {
        background: rgba(210,153,34,0.08);
        border: 1px solid rgba(210,153,34,0.2);
        border-left: 3px solid #d29922;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        font-size: 0.8rem;
        color: #b58900;
        margin-top: 1.5rem;
        line-height: 1.6;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060d18 0%, #0a1628 50%, #060d18 100%);
        border-right: 1px solid rgba(56,139,253,0.1);
    }

    [data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }

    .sidebar-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #388bfd, #00d2b4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: block;
        margin-bottom: 0.3rem;
    }

    .nav-section { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #3c5775 !important; margin-bottom: 0.5rem; margin-top: 1rem; }
    .nav-item    { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-radius: 8px; font-size: 0.85rem; color: #7d9db5 !important; margin-bottom: 2px; }
    .nav-item.active { background: rgba(56,139,253,0.15); color: #58a6ff !important; font-weight: 600; }

    .sidebar-info-block { background: rgba(13,27,42,0.8); border: 1px solid rgba(56,139,253,0.1); border-radius: 10px; padding: 0.8rem 1rem; margin-top: 0.5rem; font-size: 0.78rem; }
    .sidebar-info-row   { display: flex; justify-content: space-between; margin-bottom: 0.4rem; color: #5c7a9e !important; }
    .sidebar-info-val   { color: #c9d1d9 !important; font-weight: 600; }

    /* ── Streamlit metric tweak ── */
    [data-testid="stMetric"] { background: linear-gradient(145deg,#0d1b2a,#111f30) !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 14px !important; padding: 1rem 1.2rem !important; }
    [data-testid="stMetricLabel"] { color: #5c7a9e !important; font-size: 0.75rem !important; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.5rem !important; font-family: 'Space Grotesk',sans-serif !important; }
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


def format_change(value: float):
    """Return (label, css_class) tuple."""
    if value is None or pd.isna(value):
        return "N/A", "change-flat"
    if value > 0:
        return f"▲ {abs(value):.2f}%", "change-up"
    if value < 0:
        return f"▼ {abs(value):.2f}%", "change-down"
    return "0.00%", "change-flat"


# ── Home Page ──────────────────────────────────────────────────────────────

def render_home():
    now_str = datetime.now().strftime("%d %b %Y · %I:%M %p IST")

    # ── Hero section ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-badge">⚡ AI-Powered &nbsp;·&nbsp; NSE &nbsp;·&nbsp; NIFTY 50</div>
        <h1 class="hero-title">StockVision <span>AI</span></h1>
        <p class="hero-subtitle">
            Institutional-grade market analytics, risk intelligence and ML-driven
            price forecasting for the NIFTY 50 universe — built for data-driven investors.
        </p>
        <div class="live-row">
            <div class="pulse-dot"></div>
            <span class="live-text">Live Data</span>
            &nbsp;·&nbsp; Last refreshed: {now_str}
        </div>
        <div class="hero-stats">
            <div>
                <span class="hero-stat-value">{len(ALL_TICKERS)}</span>
                <span class="hero-stat-label">Stocks Tracked</span>
            </div>
            <div>
                <span class="hero-stat-value">{len(STOCK_UNIVERSE)}</span>
                <span class="hero-stat-label">Sectors Covered</span>
            </div>
            <div>
                <span class="hero-stat-value">3</span>
                <span class="hero-stat-label">ML Models</span>
            </div>
            <div>
                <span class="hero-stat-value">5+yr</span>
                <span class="hero-stat-label">Historical Data</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI tiles ─────────────────────────────────────────────────────────
    kpi_data = [
        ("blue",   "📊", str(len(ALL_TICKERS)), "Stocks Monitored",    "NIFTY 50 Universe"),
        ("green",  "🏭", str(len(STOCK_UNIVERSE)), "Sectors Covered",  "IT · Banking · Energy · Auto"),
        ("teal",   "🤖", "3",                   "ML Models Deployed",  "XGBoost · RF · Logistic"),
        ("purple", "📅", "5+ yrs",              "Historical Data",     "Daily OHLCV via yfinance"),
    ]

    kpi_cols = st.columns(4)
    for col, (color, icon, value, label, sub) in zip(kpi_cols, kpi_data):
        col.markdown(f"""
        <div class="kpi-card {color}">
            <span class="kpi-icon">{icon}</span>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom:2rem"></div>', unsafe_allow_html=True)

    # ── Market snapshot ───────────────────────────────────────────────────
    st.markdown('<p class="section-header-text">📡 Market Snapshot</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    snap_cols = st.columns(len(ALL_TICKERS[:6]))
    for col, ticker in zip(snap_cols, ALL_TICKERS[:6]):
        df = load_prices(ticker, days=5)
        short = ticker.replace(".NS", "")
        cname = COMPANY_INFO.get(ticker, {}).get("name", ticker)

        if df.empty or len(df) < 2:
            col.markdown(f"""
            <div class="ticker-card">
                <div class="ticker-symbol">{short}</div>
                <div class="ticker-name">{cname[:20]}</div>
                <div class="ticker-price">—</div>
                <span class="ticker-change change-flat">N/A</span>
            </div>""", unsafe_allow_html=True)
            continue

        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        chg    = (latest["close_price"] - prev["close_price"]) / prev["close_price"] * 100
        chg_str, chg_cls = format_change(chg)

        col.markdown(f"""
        <div class="ticker-card">
            <div class="ticker-symbol">{short}</div>
            <div class="ticker-name">{cname[:22]}</div>
            <div class="ticker-price">₹{latest['close_price']:,.2f}</div>
            <span class="ticker-change {chg_cls}">{chg_str}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom:2rem"></div>', unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────
    chart_col, side_col = st.columns([3, 2], gap="large")

    with chart_col:
        st.markdown('<p class="section-header-text">📈 NIFTY 50 — 1 Year Performance</p>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

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

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=nifty_df["trade_date"],
                y=nifty_df["close_price"],
                mode="lines",
                name="NIFTY 50",
                line=dict(color="#388bfd", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(56,139,253,0.07)",
                hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra></extra>",
            ))

            if len(nifty_df) > 50:
                nifty_df = nifty_df.copy()
                nifty_df["ma50"] = nifty_df["close_price"].rolling(50).mean()
                fig.add_trace(go.Scatter(
                    x=nifty_df["trade_date"],
                    y=nifty_df["ma50"],
                    mode="lines",
                    name="50-Day MA",
                    line=dict(color="#00d2b4", width=1.5, dash="dot"),
                    hovertemplate="<b>50-Day MA</b><br>₹%{y:,.2f}<extra></extra>",
                ))

            fig.update_layout(
                template="plotly_dark",
                height=340,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color="#3c5775", tickformat="%b '%y"),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.04)",
                    color="#3c5775",
                    tickprefix="₹",
                    tickformat=",.0f",
                ),
                legend=dict(
                    orientation="h", yanchor="top", y=1.12,
                    xanchor="right", x=1,
                    font=dict(size=11, color="#8b9fbe"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ Run the ingestion pipeline to load market data.")

    with side_col:
        st.markdown('<p class="section-header-text">🏭 Sector Distribution</p>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        sectors = {}
        for ticker, info in COMPANY_INFO.items():
            sec = info.get("sector", "Other")
            if sec != "Benchmark":
                sectors[sec] = sectors.get(sec, 0) + 1

        sec_df = pd.DataFrame(list(sectors.items()), columns=["Sector", "Stocks"])
        sec_df = sec_df.sort_values("Stocks", ascending=True)

        bar_colors = ["#388bfd", "#3fb950", "#00d2b4", "#bc8cff"]
        fig2 = go.Figure()
        for idx, (_, row) in enumerate(sec_df.iterrows()):
            fig2.add_trace(go.Bar(
                x=[row["Stocks"]],
                y=[row["Sector"]],
                orientation="h",
                marker_color=bar_colors[idx % len(bar_colors)],
                marker_line_width=0,
                showlegend=False,
                hovertemplate=f"<b>{row['Sector']}</b><br>{row['Stocks']} stocks<extra></extra>",
            ))

        fig2.update_layout(
            template="plotly_dark",
            height=220,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#3c5775", dtick=1),
            yaxis=dict(showgrid=False, color="#8b9fbe"),
            barmode="overlay",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="section-header-text" style="margin-top:1rem">🔮 Latest AI Forecasts</p>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

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
    st.markdown('<div class="h-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header-text">🚀 Explore the Platform</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    features = [
        ("fi-blue",   "tag-blue",   "📊", "Stock Explorer",
         "Interactive candlestick charts, RSI, MACD, Bollinger Bands and full technical analysis suite.",
         "Charts & Indicators"),
        ("fi-green",  "tag-green",  "🔮", "Forecast Centre",
         "ML-powered next-day and 5-day price direction predictions with confidence scores.",
         "AI Predictions"),
        ("fi-teal",   "tag-teal",   "🧪", "Model Laboratory",
         "Compare model performance, leaderboard rankings, feature importance and evaluation metrics.",
         "ML Insights"),
        ("fi-purple", "tag-purple", "🔍", "Data Quality",
         "Monitor pipeline health, data freshness, missing records and ingestion integrity.",
         "Pipeline Health"),
    ]

    fc = st.columns(4, gap="medium")
    for col, (fi_cls, tag_cls, icon, title, desc, tag) in zip(fc, features):
        col.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon {fi_cls}">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
            <span class="feature-tag {tag_cls}">{tag}</span>
        </div>""", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer-bar">
        ⚠️ <strong>Important Disclaimer:</strong> All forecasts and analytics are generated
        from historical market data for <strong>educational and research purposes only</strong>.
        They do not constitute financial or investment advice. Past market patterns do not
        guarantee future performance. Always consult a qualified financial advisor before
        making investment decisions. StockVision AI is a portfolio project.
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <span class="sidebar-logo">📈 StockVision AI</span>
    <span style="font-size:0.72rem;color:#3c5775;text-transform:uppercase;letter-spacing:0.8px;">
        Market Analytics Platform
    </span>
    """, unsafe_allow_html=True)

    st.markdown('<div class="h-divider" style="margin:0.8rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)

    nav_items = [
        ("🏠", "Home",             True),
        ("📊", "Stock Explorer",   False),
        ("🔮", "Forecast Centre",  False),
        ("🧪", "Model Laboratory", False),
        ("🔍", "Data Quality",     False),
    ]
    for icon, label, active in nav_items:
        cls = "nav-item active" if active else "nav-item"
        st.markdown(f'<div class="{cls}">{icon}&nbsp;&nbsp;{label}</div>', unsafe_allow_html=True)

    st.markdown('<div class="h-divider" style="margin:0.8rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-section">Platform Info</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-info-block">
        <div class="sidebar-info-row"><span>Universe</span><span class="sidebar-info-val">NIFTY 50</span></div>
        <div class="sidebar-info-row"><span>Stocks</span><span class="sidebar-info-val">{len(ALL_TICKERS)} tracked</span></div>
        <div class="sidebar-info-row"><span>Forecast</span><span class="sidebar-info-val">1-day &amp; 5-day</span></div>
        <div class="sidebar-info-row"><span>Models</span><span class="sidebar-info-val">XGB · RF · LR</span></div>
        <div class="sidebar-info-row" style="margin-bottom:0"><span>Data</span><span class="sidebar-info-val">yfinance</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="h-divider" style="margin:0.8rem 0;"></div>', unsafe_allow_html=True)
    st.caption("Built for Data Analyst Portfolio · 2024\nPowered by XGBoost · Streamlit · Plotly")



# ── Render ─────────────────────────────────────────────────────────────────
render_home()
