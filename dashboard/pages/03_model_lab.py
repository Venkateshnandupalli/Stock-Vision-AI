"""
StockVision AI — Model Laboratory Page (Premium 3D Edition)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Model Lab | StockVision AI", page_icon="🧪", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Space+Grotesk:wght@400;500;600;700');

    html, body { background: #030712 !important; }
[data-testid="stApp"], [data-testid="stAppViewContainer"] { background: #030712 !important; }
[data-testid="stAppViewBlockContainer"] { animation: pageFadeIn 0.35s ease; }
@keyframes pageFadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background: #030712; color: #e2e8f0; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem !important; max-width: 1440px; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #030712, #060e1c) !important;
        border-right: 1px solid rgba(0,255,136,0.08) !important;
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
        background: rgba(0,255,136,0.05) !important;
        border-color: rgba(0,255,136,0.12) !important;
        color: #00ff88 !important;
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: rgba(0,255,136,0.08) !important;
        border-color: rgba(0,255,136,0.2) !important;
        color: #00ff88 !important; font-weight: 600 !important;
        box-shadow: 0 0 18px rgba(0,255,136,0.07) !important;
    }
    [data-testid="stSidebarNav"] span { color: inherit !important; font-weight: inherit !important; }

    .page-header {
        background: radial-gradient(ellipse at 60% 30%, rgba(0,255,136,0.08) 0%, transparent 55%),
                    rgba(255,255,255,0.02);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0,255,136,0.1);
        border-radius: 20px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
    }

    .page-header::before {
        content: '';
        position: absolute; inset: 0;
        background-image:
            linear-gradient(rgba(0,255,136,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,255,136,0.025) 1px, transparent 1px);
        background-size: 32px 32px; pointer-events: none;
    }

    .page-title {
        font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, #00ff88, #00f5ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        filter: drop-shadow(0 0 12px rgba(0,255,136,0.25)); margin: 0 0 0.3rem 0;
    }

    .page-subtitle { font-size: 0.85rem; color: #3d5268; }

    .sec-label {
        font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700;
        color: #e2e8f0; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 8px;
    }
    .sec-line { height: 1px; flex: 1; background: linear-gradient(90deg, rgba(0,255,136,0.35), transparent); }

    /* Neon leaderboard */
    .leaderboard-wrap {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(0,255,136,0.1);
        border-radius: 16px; overflow: hidden;
    }

    .lb-row {
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
        padding: 12px 16px;
        font-size: 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.03);
        transition: background 0.2s ease;
    }

    .lb-row:hover { background: rgba(0,255,136,0.04); }
    .lb-row.header { background: rgba(0,255,136,0.06); font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #2d4560; }

    .lb-model { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #00ff88; font-size: 0.78rem; }
    .lb-val   { font-family: 'JetBrains Mono', monospace; color: #7d8fa8; font-size: 0.78rem; }
    .lb-rank-1 .lb-model { color: #f59e0b; text-shadow: 0 0 10px rgba(245,158,11,0.4); }
    .lb-rank-2 .lb-model { color: #94a3b8; }
    .lb-rank-3 .lb-model { color: #b45309; }

    .filter-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1.2rem;
    }

    .note-card {
        background: rgba(0,255,136,0.04);
        border: 1px solid rgba(0,255,136,0.12);
        border-left: 3px solid #00ff88;
        border-radius: 12px; padding: 1rem 1.2rem;
        font-size: 0.8rem; color: #3d7a5a; line-height: 1.7; margin-top: 1rem;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.025) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important; padding: 1rem !important;
    }
    [data-testid="stMetricLabel"] { color: #4a6080 !important; font-size: 0.72rem !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-family: 'JetBrains Mono', monospace !important; }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#00ff88,#00f5ff); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown("""
<div class="page-header">
    <div class="page-title">🧪 Model Laboratory</div>
    <div class="page-subtitle">3D model comparison scatter, leaderboard rankings, feature importance and evaluation metrics.</div>
</div>
""", unsafe_allow_html=True)

# ── Load metrics ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_metrics():
    try:
        from src.database.queries import get_model_metrics
        return get_model_metrics()
    except Exception:
        return pd.DataFrame()

metrics_df = load_metrics()

# Demo data if empty
if metrics_df.empty:
    st.info("No model metrics found yet. Using demo data — run training to populate with real results.")
    metrics_df = pd.DataFrame({
        "model_name":            ["xgboost_regressor", "random_forest", "gradient_boosting", "ridge_regression", "naive_baseline"],
        "ticker":                ["TCS.NS"] * 5,
        "target":                ["target_return_1d"] * 5,
        "mae":                   [0.0082, 0.0091, 0.0089, 0.0098, 0.0110],
        "rmse":                  [0.0121, 0.0134, 0.0129, 0.0142, 0.0158],
        "r_squared":             [0.42,   0.31,   0.35,   0.22,   0.0],
        "directional_accuracy":  [0.542,  0.528,  0.531,  0.514,  0.500],
        "improvement_over_baseline": [0.255, 0.173, 0.191, 0.109, 0.0],
    })

# ── Filter controls ────────────────────────────────────────────────────────
col_filter1, col_filter2 = st.columns(2)
with col_filter1:
    if "ticker" in metrics_df.columns:
        tickers      = ["All"] + list(metrics_df["ticker"].dropna().unique())
        sel_ticker   = st.selectbox("Filter by Ticker", tickers)
    else:
        sel_ticker = "All"

with col_filter2:
    if "target" in metrics_df.columns:
        targets    = ["All"] + list(metrics_df["target"].dropna().unique())
        sel_target = st.selectbox("Filter by Target", targets)
    else:
        sel_target = "All"

filtered = metrics_df.copy()
if sel_ticker != "All" and "ticker" in filtered.columns:
    filtered = filtered[filtered["ticker"] == sel_ticker]
if sel_target != "All" and "target" in filtered.columns:
    filtered = filtered[filtered["target"] == sel_target]

# ── 3D Model Comparison Scatter ────────────────────────────────────────────
st.markdown("""
<div class="sec-label">
    🌐 3D Model Comparison — MAE × Directional Accuracy × R²
    <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

if "mae" in filtered.columns and "directional_accuracy" in filtered.columns:
    agg = filtered.groupby("model_name").agg(
        mae=("mae", "mean"),
        dir_acc=("directional_accuracy", "mean"),
        r2=("r_squared", "mean") if "r_squared" in filtered.columns else ("mae", "count"),
    ).reset_index()

    neon_palette = ["#00f5ff", "#00ff88", "#7c3aed", "#f59e0b", "#ff4d6a"]
    fig_3d = go.Figure()

    for i, row in agg.iterrows():
        color = neon_palette[i % len(neon_palette)]
        fig_3d.add_trace(go.Scatter3d(
            x=[row["mae"]],
            y=[row["dir_acc"]],
            z=[row.get("r2", 0)],
            mode="markers+text",
            name=row["model_name"],
            text=[row["model_name"].replace("_", " ").title()],
            textposition="top center",
            textfont=dict(color=color, size=10),
            marker=dict(
                size=14,
                color=color,
                opacity=0.9,
                symbol="circle",
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
            hovertemplate=(
                f"<b>{row['model_name']}</b><br>"
                f"MAE: {row['mae']:.4f}<br>"
                f"Dir Acc: {row['dir_acc']:.1%}<br>"
                f"R²: {row.get('r2',0):.3f}<extra></extra>"
            ),
        ))

        # Drop line to floor
        fig_3d.add_trace(go.Scatter3d(
            x=[row["mae"], row["mae"]],
            y=[row["dir_acc"], row["dir_acc"]],
            z=[0, row.get("r2", 0)],
            mode="lines",
            line=dict(color=color, width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title="MAE (lower=better)", color="#4a6080", showgrid=True, gridcolor="rgba(0,255,136,0.05)"),
            yaxis=dict(title="Directional Accuracy", color="#4a6080", showgrid=True, gridcolor="rgba(0,255,136,0.05)", tickformat=".0%"),
            zaxis=dict(title="R² Score", color="#4a6080", showgrid=True, gridcolor="rgba(0,255,136,0.05)"),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
        ),
        height=480, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#7d8fa8", size=10), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ── Neon Leaderboard ───────────────────────────────────────────────────────
st.markdown("""
<div class="sec-label">
    🏆 Model Leaderboard
    <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

display_cols = ["model_name", "mae", "rmse", "r_squared", "directional_accuracy", "improvement_over_baseline"]
existing_cols = [c for c in display_cols if c in filtered.columns]
if not filtered.empty:
    lb_df = filtered[existing_cols].sort_values("mae").reset_index(drop=True)

    # Build neon HTML table
    rank_classes = ["lb-rank-1", "lb-rank-2", "lb-rank-3"]
    header = """
    <div class="leaderboard-wrap">
        <div class="lb-row header">
            <div>Model</div><div>MAE ↓</div><div>RMSE</div><div>R²</div><div>Dir Acc</div>
        </div>
    """
    rows_html = ""
    for i, row in lb_df.iterrows():
        rank_cls = rank_classes[i] if i < 3 else ""
        rank_badge = ["🥇 ", "🥈 ", "🥉 "][i] if i < 3 else f"{i+1}. "
        mae  = f"{row['mae']:.4f}"  if 'mae'  in row else "—"
        rmse = f"{row['rmse']:.4f}" if 'rmse' in row else "—"
        r2   = f"{row['r_squared']:.3f}" if 'r_squared' in row else "—"
        da   = f"{row['directional_accuracy']:.1%}" if 'directional_accuracy' in row else "—"
        model_name = str(row['model_name']).replace("_", " ").title()
        rows_html += f"""
        <div class="lb-row {rank_cls}">
            <div class="lb-model">{rank_badge}{model_name}</div>
            <div class="lb-val">{mae}</div>
            <div class="lb-val">{rmse}</div>
            <div class="lb-val">{r2}</div>
            <div class="lb-val">{da}</div>
        </div>
        """
    st.markdown(header + rows_html + "</div>", unsafe_allow_html=True)

# ── MAE + Directional Accuracy charts ─────────────────────────────────────
chart_l, chart_r = st.columns(2, gap="large")

with chart_l:
    st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-label">
        📊 MAE by Model
        <div class="sec-line"></div>
    </div>
    """, unsafe_allow_html=True)

    if "model_name" in filtered.columns and "mae" in filtered.columns:
        mae_agg = filtered.groupby("model_name")["mae"].mean().reset_index().sort_values("mae")
        neon_palette = ["#00ff88", "#00f5ff", "#7c3aed", "#f59e0b", "#ff4d6a"]
        colors = [neon_palette[i % len(neon_palette)] for i in range(len(mae_agg))]

        fig_mae = go.Figure()
        for i, (_, row) in enumerate(mae_agg.iterrows()):
            c = colors[i]
            fig_mae.add_trace(go.Bar(
                x=[row["mae"]], y=[row["model_name"].replace("_", " ").title()],
                orientation="h",
                marker=dict(color=c, opacity=0.85, line=dict(color=c, width=0)),
                showlegend=False,
                hovertemplate=f"<b>{row['model_name']}</b><br>MAE: {row['mae']:.4f}<extra></extra>",
            ))
            fig_mae.add_trace(go.Bar(
                x=[row["mae"]], y=[row["model_name"].replace("_", " ").title()],
                orientation="h",
                marker=dict(color=c, opacity=0.12, line=dict(color=c, width=6)),
                showlegend=False, hoverinfo="skip",
            ))

        fig_mae.update_layout(
            template="plotly_dark", height=280, barmode="overlay",
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(0,255,136,0.05)", color="#4a6080"),
            yaxis=dict(showgrid=False, color="#7d8fa8"),
        )
        st.plotly_chart(fig_mae, use_container_width=True)

with chart_r:
    st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-label">
        🎯 Directional Accuracy
        <div class="sec-line"></div>
    </div>
    """, unsafe_allow_html=True)

    if "directional_accuracy" in filtered.columns:
        da_agg = filtered.groupby("model_name")["directional_accuracy"].mean().reset_index()
        neon_palette = ["#00f5ff", "#00ff88", "#7c3aed", "#f59e0b", "#ff4d6a"]

        fig_da = go.Figure()
        for i, (_, row) in enumerate(da_agg.iterrows()):
            c = neon_palette[i % len(neon_palette)]
            fig_da.add_trace(go.Bar(
                x=[row["model_name"].replace("_", " ").title()], y=[row["directional_accuracy"]],
                marker=dict(color=c, opacity=0.85),
                showlegend=False,
                hovertemplate=f"<b>{row['model_name']}</b><br>Dir Acc: {row['directional_accuracy']:.1%}<extra></extra>",
            ))

        fig_da.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,77,106,0.5)",
                         annotation_text="Random Baseline", annotation_font_color="#ff4d6a")
        fig_da.update_layout(
            template="plotly_dark", height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#7d8fa8"),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,255,136,0.05)", color="#4a6080",
                       tickformat=".0%", range=[0.45, 0.62]),
        )
        st.plotly_chart(fig_da, use_container_width=True)

# ── Interpretation note ────────────────────────────────────────────────────
st.markdown("""
<div class="note-card">
    <strong>📖 Interpretation Guide:</strong><br>
    • <strong>MAE</strong>: Mean Absolute Error — lower is better (measures average prediction error magnitude)<br>
    • <strong>Directional Accuracy</strong>: % of times the model correctly predicted up/down direction — &gt;55% is considered meaningful<br>
    • <strong>R²</strong>: Coefficient of determination — how much variance the model explains (0–1, higher is better)<br>
    • <strong>vs Baseline</strong>: % improvement over naive random-walk baseline
</div>
""", unsafe_allow_html=True)
