"""
StockVision AI — Data Quality Page
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
from datetime import datetime

from src.utils.config import ALL_TICKERS, COMPANY_INFO

st.set_page_config(page_title="Data Quality | StockVision AI", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f2027 0%, #1a2f3e 100%); }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Data Quality Dashboard")
st.caption("Monitor pipeline health, data freshness, and ingestion status.")

# ── Pipeline logs ──────────────────────────────────────────────────────────
st.subheader("🔄 Pipeline Execution History")

@st.cache_data(ttl=60)
def load_pipeline_logs():
    try:
        from src.database.connection import get_session
        from sqlalchemy import text
        with get_session() as s:
            result = s.execute(text(
                "SELECT * FROM vw_pipeline_health ORDER BY run_at DESC LIMIT 20"
            ))
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception:
        return pd.DataFrame()

logs_df = load_pipeline_logs()

if logs_df.empty:
    st.info("No pipeline logs found. Run the ingestion pipeline:\n```\npython -m src.ingestion.fetch_market_data\n```")
else:
    # Status indicator
    last_status = logs_df.iloc[0]["status"] if not logs_df.empty else "unknown"
    status_color = {"success": "🟢", "partial": "🟡", "error": "🔴"}.get(last_status, "⚪")
    st.markdown(f"**Pipeline Status:** {status_color} {last_status.upper()}")
    st.dataframe(logs_df, hide_index=True, use_container_width=True)

st.markdown("---")

# ── Data coverage per ticker ────────────────────────────────────────────────
st.subheader("📊 Data Coverage by Ticker")

@st.cache_data(ttl=300)
def load_coverage():
    try:
        from src.database.connection import get_session
        from sqlalchemy import text
        with get_session() as s:
            result = s.execute(text("""
                SELECT
                    c.ticker,
                    c.company_name,
                    c.sector,
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

coverage_df = load_coverage()

if coverage_df.empty:
    # Show placeholder
    placeholder = pd.DataFrame({
        "Ticker":       ALL_TICKERS,
        "Company":      [COMPANY_INFO.get(t, {}).get("name", t) for t in ALL_TICKERS],
        "Sector":       [COMPANY_INFO.get(t, {}).get("sector", "-") for t in ALL_TICKERS],
        "Status":       ["⚪ No data" for _ in ALL_TICKERS],
    })
    st.dataframe(placeholder, hide_index=True, use_container_width=True)
    st.warning("No data in database. Run: `python -m src.ingestion.fetch_market_data`")
else:
    # Add freshness column
    if "date_end" in coverage_df.columns:
        coverage_df["date_end"] = pd.to_datetime(coverage_df["date_end"])
        coverage_df["days_since_update"] = (
            pd.Timestamp.today() - coverage_df["date_end"]
        ).dt.days
        coverage_df["freshness"] = coverage_df["days_since_update"].apply(
            lambda d: "🟢 Fresh" if d <= 2 else ("🟡 Stale" if d <= 7 else "🔴 Old")
        )
    st.dataframe(coverage_df, hide_index=True, use_container_width=True)

st.markdown("---")

# ── Quick actions ──────────────────────────────────────────────────────────
st.subheader("⚡ Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📥 Run Ingestion**")
    st.code("python -m src.ingestion.fetch_market_data", language="bash")

with col2:
    st.markdown("**⚙️ Build Features**")
    st.code("python -m src.features.build_features", language="bash")

with col3:
    st.markdown("**🤖 Train Models**")
    st.code("python -m src.models.train", language="bash")

st.markdown("---")
st.caption(f"Dashboard generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
