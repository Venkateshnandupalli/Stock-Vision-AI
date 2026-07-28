"""
StockVision AI — Data Quality Dashboard (Premium 3D Edition)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

from src.utils.config import ALL_TICKERS, COMPANY_INFO

st.set_page_config(page_title="Data Quality | StockVision AI", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #030712; color: #e2e8f0; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem !important; max-width: 1440px; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #030712, #060e1c) !important;
        border-right: 1px solid rgba(245,158,11,0.08) !important;
    }

    .page-header {
        background: radial-gradient(ellipse at 75% 30%, rgba(245,158,11,0.08) 0%, transparent 55%),
                    rgba(255,255,255,0.02);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(245,158,11,0.12);
        border-radius: 20px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
    }

    .page-header::before {
        content: '';
        position: absolute; inset: 0;
        background-image:
            linear-gradient(rgba(245,158,11,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(245,158,11,0.02) 1px, transparent 1px);
        background-size: 32px 32px; pointer-events: none;
    }

    .page-title {
        font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        filter: drop-shadow(0 0 12px rgba(245,158,11,0.25)); margin: 0 0 0.3rem 0;
    }

    .page-subtitle { font-size: 0.85rem; color: #3d5268; }

    .sec-label {
        font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
        color: #e2e8f0; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 8px;
    }
    .sec-line { height: 1px; flex: 1; background: linear-gradient(90deg, rgba(245,158,11,0.35), transparent); }

    /* Status cards */
    .status-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin-bottom: 1.5rem; }

    .status-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 1.3rem 1.5rem;
        transition: all 0.3s ease; position: relative; overflow: hidden;
    }

    .status-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    }

    .status-card.success { border-color: rgba(0,255,136,0.15); }
    .status-card.warning { border-color: rgba(245,158,11,0.15); }
    .status-card.error   { border-color: rgba(255,77,106,0.15); }
    .status-card.idle    { border-color: rgba(125,143,168,0.12); }

    .status-dot {
        width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 8px;
    }

    .dot-success { background: #00ff88; box-shadow: 0 0 10px rgba(0,255,136,0.5); animation: blink 2s infinite; }
    .dot-warning { background: #f59e0b; box-shadow: 0 0 10px rgba(245,158,11,0.5); }
    .dot-error   { background: #ff4d6a; box-shadow: 0 0 10px rgba(255,77,106,0.5); }
    .dot-idle    { background: #7d8fa8; }

    @keyframes blink {
        0%,100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    .status-label { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #2d4560; }
    .status-value { font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 700; color: #f8fafc; margin: 0.3rem 0 0; }

    /* Coverage table */
    .cov-table {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; overflow: hidden;
    }

    .cov-row {
        display: grid; grid-template-columns: 1fr 2fr 1fr 1fr 1fr 1fr;
        padding: 10px 16px; font-size: 0.78rem;
        border-bottom: 1px solid rgba(255,255,255,0.03);
        transition: background 0.2s;
    }

    .cov-row:hover { background: rgba(245,158,11,0.04); }
    .cov-row.header { background: rgba(245,158,11,0.06); font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #2d4560; }

    .cov-ticker { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #f59e0b; font-size: 0.76rem; }
    .cov-val    { font-family: 'JetBrains Mono', monospace; color: #7d8fa8; font-size: 0.75rem; }
    .cov-fresh  { color: #00ff88; font-weight: 700; }
    .cov-stale  { color: #f59e0b; font-weight: 700; }
    .cov-old    { color: #ff4d6a; font-weight: 700; }
    .cov-none   { color: #4a6080; }

    /* Action cards */
    .action-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px; padding: 1.3rem 1.5rem;
        transition: all 0.3s ease;
    }

    .action-card:hover {
        border-color: rgba(245,158,11,0.2);
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.4);
    }

    .action-title { font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem; }
    .action-cmd   { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #00f5ff; background: rgba(0,245,255,0.05); border: 1px solid rgba(0,245,255,0.1); border-radius: 8px; padding: 8px 12px; margin-top: 0.5rem; }

    .footer-bar { font-size: 0.72rem; color: #1e3048; text-align: center; margin-top: 2rem; font-family: 'JetBrains Mono', monospace; }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#f59e0b,#00f5ff); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown("""
<div class="page-header">
    <div class="page-title">🔍 Data Quality Dashboard</div>
    <div class="page-subtitle">Monitor pipeline health, data freshness, 3D coverage heatmap and ingestion status.</div>
</div>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_pipeline_logs():
    try:
        from src.database.connection import get_session
        from sqlalchemy import text
        with get_session() as s:
            result = s.execute(text("SELECT * FROM vw_pipeline_health ORDER BY run_at DESC LIMIT 20"))
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_coverage():
    try:
        from src.database.connection import get_session
        from sqlalchemy import text
        with get_session() as s:
            result = s.execute(text("""
                SELECT
                    c.ticker, c.company_name, c.sector,
                    COUNT(sp.price_id)           AS row_count,
                    MIN(sp.trade_date)           AS date_start,
                    MAX(sp.trade_date)           AS date_end,
                    SUM(CASE WHEN sp.close_price IS NULL THEN 1 ELSE 0 END) AS null_prices,
                    SUM(CASE WHEN sp.volume = 0 THEN 1 ELSE 0 END) AS zero_volume_days
                FROM companies c
                LEFT JOIN stock_prices sp ON sp.company_id = c.company_id
                WHERE c.is_active = TRUE
                GROUP BY c.company_id, c.ticker, c.company_name, c.sector
                ORDER BY c.ticker
            """))
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception:
        return pd.DataFrame()

logs_df     = load_pipeline_logs()
coverage_df = load_coverage()

# ── Pipeline Status cards ──────────────────────────────────────────────────
st.markdown("""
<div class="sec-label">
    🔄 Pipeline Status
    <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

if not logs_df.empty:
    last       = logs_df.iloc[0]
    status_map = {"success": ("dot-success", "success"), "partial": ("dot-warning", "warning"), "error": ("dot-error", "error")}
    dot_cls, card_cls = status_map.get(str(last.get("status", "")), ("dot-idle", "idle"))

    st.markdown(f"""
    <div class="status-grid">
        <div class="status-card {card_cls}">
            <div class="status-label"><span class="status-dot {dot_cls}"></span>Last Pipeline Run</div>
            <div class="status-value">{str(last.get('status','-')).upper()}</div>
        </div>
        <div class="status-card">
            <div class="status-label">Last Run At</div>
            <div class="status-value" style="font-size:0.9rem">{str(last.get('run_at','-'))[:19]}</div>
        </div>
        <div class="status-card">
            <div class="status-label">Records Processed</div>
            <div class="status-value">{last.get('records_inserted', '—')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-label">
        📋 Execution History
        <div class="sec-line"></div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(logs_df, hide_index=True, use_container_width=True)
else:
    st.markdown("""
    <div class="status-grid">
        <div class="status-card idle">
            <div class="status-label"><span class="status-dot dot-idle"></span>Pipeline Status</div>
            <div class="status-value">NO DATA</div>
        </div>
        <div class="status-card idle">
            <div class="status-label">Records</div>
            <div class="status-value">—</div>
        </div>
        <div class="status-card idle">
            <div class="status-label">Last Run</div>
            <div class="status-value">—</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("No pipeline logs. Run: `python -m src.ingestion.fetch_market_data`")

# ── 3D Coverage Heatmap ────────────────────────────────────────────────────
st.markdown("""
<div class="sec-label" style="margin-top:1rem">
    🌡️ 3D Coverage Heatmap — Row Count per Ticker
    <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

if not coverage_df.empty and "row_count" in coverage_df.columns:
    tickers_list = coverage_df["ticker"].tolist()
    row_counts   = coverage_df["row_count"].tolist()
    n            = len(tickers_list)

    # Build 3D bar chart using Scatter3d with stem lines
    x_positions  = np.arange(n)
    neon_colors  = [
        f"rgba(0,245,255,{0.6 + 0.4*(v/max(row_counts))})" for v in row_counts
    ]

    fig_heat = go.Figure()

    # 3D bar stems
    for i, (ticker, count) in enumerate(zip(tickers_list, row_counts)):
        color = neon_colors[i]
        fig_heat.add_trace(go.Scatter3d(
            x=[i, i], y=[0, 0], z=[0, count],
            mode="lines",
            line=dict(color=color, width=6),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_heat.add_trace(go.Scatter3d(
            x=[i], y=[0], z=[count],
            mode="markers+text",
            marker=dict(size=8, color=color, symbol="circle",
                        line=dict(color="white", width=1)),
            text=[ticker.replace(".NS", "")],
            textposition="top center",
            textfont=dict(color=color, size=9),
            name=ticker,
            hovertemplate=f"<b>{ticker}</b><br>Rows: {count:,}<extra></extra>",
        ))

    fig_heat.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title="Ticker", titlefont=dict(color="#4a6080")),
            yaxis=dict(showgrid=False, showticklabels=False, title=""),
            zaxis=dict(
                showgrid=True, gridcolor="rgba(245,158,11,0.06)",
                color="#4a6080", title="Row Count",
                titlefont=dict(color="#f59e0b"), tickformat=",",
            ),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=2.0, y=-0.5, z=1.2)),
        ),
        height=450, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    # Generate demo 3D chart
    demo_tickers = [t.replace(".NS","") for t in ALL_TICKERS]
    demo_counts  = np.random.randint(800, 1800, len(demo_tickers))

    fig_demo = go.Figure()
    for i, (t, c) in enumerate(zip(demo_tickers, demo_counts)):
        color = f"rgba(245,158,11,{0.4 + 0.6*(c/demo_counts.max())})"
        fig_demo.add_trace(go.Scatter3d(
            x=[i,i], y=[0,0], z=[0,c], mode="lines",
            line=dict(color=color, width=6), showlegend=False, hoverinfo="skip",
        ))
        fig_demo.add_trace(go.Scatter3d(
            x=[i], y=[0], z=[c], mode="markers+text",
            marker=dict(size=8, color=color), text=[t],
            textposition="top center", textfont=dict(color=color, size=9),
            showlegend=False, hovertemplate=f"<b>{t}</b><br>~{c:,} rows (demo)<extra></extra>",
        ))

    fig_demo.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title="Ticker"),
            yaxis=dict(showgrid=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=True, gridcolor="rgba(245,158,11,0.06)", color="#4a6080",
                       title="Row Count", tickformat=","),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=2.0, y=-0.5, z=1.2)),
        ),
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    st.plotly_chart(fig_demo, use_container_width=True)

# ── Coverage Table ─────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-label">
    📊 Data Coverage by Ticker
    <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

if not coverage_df.empty:
    coverage_df["date_end"] = pd.to_datetime(coverage_df["date_end"])
    coverage_df["days_since"] = (pd.Timestamp.today() - coverage_df["date_end"]).dt.days
    coverage_df["freshness"] = coverage_df["days_since"].apply(
        lambda d: "🟢 Fresh" if d <= 2 else ("🟡 Stale" if d <= 7 else "🔴 Old")
    )

    header = """
    <div class="cov-table">
        <div class="cov-row header">
            <div>Ticker</div><div>Company</div><div>Sector</div><div>Rows</div><div>Latest Date</div><div>Freshness</div>
        </div>
    """
    rows_html = ""
    for _, row in coverage_df.iterrows():
        fresh_cls = "cov-fresh" if "Fresh" in str(row.get("freshness","")) else ("cov-stale" if "Stale" in str(row.get("freshness","")) else "cov-old")
        rows_html += f"""
        <div class="cov-row">
            <div class="cov-ticker">{row.get('ticker','-')}</div>
            <div class="cov-val">{str(row.get('company_name','-'))[:28]}</div>
            <div class="cov-val">{row.get('sector','-')}</div>
            <div class="cov-val">{int(row.get('row_count',0)):,}</div>
            <div class="cov-val">{str(row.get('date_end','-'))[:10]}</div>
            <div class="{fresh_cls}">{row.get('freshness','—')}</div>
        </div>
        """
    st.markdown(header + rows_html + "</div>", unsafe_allow_html=True)
else:
    placeholder = pd.DataFrame({
        "Ticker":  ALL_TICKERS,
        "Company": [COMPANY_INFO.get(t, {}).get("name", t) for t in ALL_TICKERS],
        "Sector":  [COMPANY_INFO.get(t, {}).get("sector", "-") for t in ALL_TICKERS],
        "Status":  ["⚪ No data" for _ in ALL_TICKERS],
    })
    st.dataframe(placeholder, hide_index=True, use_container_width=True)
    st.warning("No data in database. Run: `python -m src.ingestion.fetch_market_data`")

# ── Quick Actions ──────────────────────────────────────────────────────────
st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec-label">
    ⚡ Quick Actions
    <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

a1, a2, a3 = st.columns(3, gap="medium")

with a1:
    st.markdown("""
    <div class="action-card">
        <div class="action-title">📥 Run Ingestion</div>
        <div style="font-size:0.78rem;color:#4a6080;margin-bottom:0.5rem">Fetch latest market data from yfinance</div>
        <div class="action-cmd">python -m src.ingestion.fetch_market_data</div>
    </div>
    """, unsafe_allow_html=True)

with a2:
    st.markdown("""
    <div class="action-card">
        <div class="action-title">⚙️ Build Features</div>
        <div style="font-size:0.78rem;color:#4a6080;margin-bottom:0.5rem">Engineer 50+ technical indicators</div>
        <div class="action-cmd">python -m src.features.build_features</div>
    </div>
    """, unsafe_allow_html=True)

with a3:
    st.markdown("""
    <div class="action-card">
        <div class="action-title">🤖 Train Models</div>
        <div style="font-size:0.78rem;color:#4a6080;margin-bottom:0.5rem">Train XGBoost, RF and regression models</div>
        <div class="action-cmd">python -m src.models.train</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f'<div class="footer-bar">StockVision AI · Dashboard generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} · For educational use only</div>', unsafe_allow_html=True)
