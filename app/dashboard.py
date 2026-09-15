"""
app/dashboard.py
----------------
Streamlit dashboard for the Volatility Regime Classifier.

Run
---
    streamlit run app/dashboard.py
"""

import os
import sys
from pathlib import Path

# Allow imports from project root when launched from any working directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Volatility Regime Classifier",
    page_icon="📉",
    layout="wide",
)

REGIME_COLORS: dict[str, str] = {
    "Low Vol": "#4CAF50",
    "Mid Vol": "#FF9800",
    "High Vol": "#F44336",
}

DATA_PATH = ROOT_DIR / "data" / "labeled_data.csv"


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.warning("⚠️ `data/labeled_data.csv` not found. Attempting to generate it now...")
        try:
            with st.spinner("Running full pipeline (downloading data, computing features, GARCH, HMM)..."):
                from run import main as run_pipeline
                run_pipeline()
        except Exception as exc:
            st.error(
                f"❌ `data/labeled_data.csv` not found and auto-generation failed: {exc}. "
                "Please run `python run.py` locally and commit `data/labeled_data.csv`."
            )
            st.stop()
    if not DATA_PATH.exists():
        st.error(
            "❌ `data/labeled_data.csv` not found. "
            "Please run `python run.py` to generate it."
        )
        st.stop()
    return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)


df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    lookback = st.slider(
        "Lookback (days)",
        min_value=252,
        max_value=len(df),
        value=min(1260, len(df)),
        step=63,
    )
    show_vol = st.checkbox("Show GARCH Volatility", value=True)
    regime_filter = st.multiselect(
        "Filter Regimes",
        options=list(REGIME_COLORS.keys()),
        default=list(REGIME_COLORS.keys()),
    )

df_view = df.tail(lookback)
if regime_filter:
    df_view = df_view[df_view["regime_name"].isin(regime_filter)]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📉 Volatility Regime Classifier — Nifty 50")
st.caption("HMM-based regime detection with GARCH(1,1) conditional volatility and SHAP explainability.")

# ── KPI cards ─────────────────────────────────────────────────────────────────
latest = df.iloc[-1]
days_in_regime = int(
    (df["regime"] == latest["regime"]).iloc[::-1].cumprod().sum()
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Regime", latest["regime_name"])
col2.metric("GARCH Vol (Ann.)", f"{latest['garch_vol'] * 100:.1f}%")
col3.metric(
    "Nifty 50",
    f"₹{latest['Close']:,.0f}",
    delta=f"{latest['returns'] * 100:.2f}%",
)
col4.metric("Days in Current Regime", days_in_regime)

st.divider()

# ── Price chart with regime shading ───────────────────────────────────────────
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_view.index,
        y=df_view["Close"],
        name="Nifty 50",
        line=dict(color="black", width=1.2),
        yaxis="y1",
    )
)

for regime_name, color in REGIME_COLORS.items():
    if regime_name not in regime_filter:
        continue
    mask = df_view["regime_name"] == regime_name
    if not mask.any():
        continue
    idx = df_view.index[mask].tolist()
    price_max = float(df_view["Close"].max())
    price_min = float(df_view["Close"].min())
    fig.add_trace(
        go.Scatter(
            x=idx + idx[::-1],
            y=[price_max] * len(idx) + [price_min] * len(idx),
            fill="toself",
            fillcolor=color,
            opacity=0.15,
            line=dict(width=0),
            name=regime_name,
            showlegend=True,
        )
    )

if show_vol:
    fig.add_trace(
        go.Scatter(
            x=df_view.index,
            y=df_view["garch_vol"] * 100,
            name="GARCH Vol (%)",
            line=dict(color="#E91E63", width=1, dash="dot"),
            yaxis="y2",
        )
    )

fig.update_layout(
    title="Nifty 50 — Price with Volatility Regimes",
    yaxis=dict(title="Price (₹)"),
    yaxis2=dict(title="Ann. Vol (%)", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", y=-0.15),
    height=450,
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── Regime distribution + statistics ─────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    dist = df_view["regime_name"].value_counts().reset_index()
    dist.columns = ["Regime", "Days"]
    fig2 = px.pie(
        dist,
        names="Regime",
        values="Days",
        color="Regime",
        color_discrete_map=REGIME_COLORS,
        title="Regime Distribution (selected period)",
        hole=0.4,
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    stats = (
        df_view.groupby("regime_name")
        .agg(
            Days=("regime", "count"),
            Mean_Vol_pct=("garch_vol", lambda x: round(x.mean() * 100, 2)),
            Mean_Ret_pct=("returns", lambda x: round(x.mean() * 100, 4)),
            Ret_Std_pct=("returns", lambda x: round(x.std() * 100, 4)),
        )
        .reset_index()
        .rename(columns={"regime_name": "Regime"})
    )
    st.markdown("#### Regime Statistics")
    st.dataframe(stats, use_container_width=True, hide_index=True)

# ── Vol distribution by regime ────────────────────────────────────────────────
fig3 = px.violin(
    df_view,
    x="regime_name",
    y=df_view["garch_vol"] * 100,
    color="regime_name",
    color_discrete_map=REGIME_COLORS,
    box=True,
    points="outliers",
    title="GARCH Volatility Distribution by Regime",
    labels={"regime_name": "Regime", "y": "Ann. Vol (%)"},
)
st.plotly_chart(fig3, use_container_width=True)

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("📋 View Raw Data (last 100 rows)"):
    show_cols = ["Close", "returns", "garch_vol", "rv_21d", "regime_name"]
    st.dataframe(df_view[show_cols].tail(100).round(4), use_container_width=True)