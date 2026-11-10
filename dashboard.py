"""Streamlit decision-support dashboard. Run after `retailiq --out artifacts`."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="RetailIQ | Demand Intelligence", page_icon="📦", layout="wide")
st.title("RetailIQ · Retail Demand Intelligence")
st.caption("Demonstration on synthetic retail data · one-step-ahead backtest · hypothetical inventory costs")
root = Path(__file__).resolve().parent / "artifacts"
required = ["test_predictions.csv", "inventory_decisions.csv", "metrics.json", "inventory_summary.json", "leaderboard.csv"]
if any(not (root / x).exists() for x in required):
    st.warning("Generate project outputs first: `pip install -e .` then `retailiq --out artifacts`.")
    st.stop()

@st.cache_data
def load():
    preds = pd.read_csv(root / "test_predictions.csv", parse_dates=["date"])
    decisions = pd.read_csv(root / "inventory_decisions.csv", parse_dates=["date"])
    leader = pd.read_csv(root / "leaderboard.csv")
    return preds, decisions, leader, json.loads((root / "metrics.json").read_text()), json.loads((root / "inventory_summary.json").read_text())

preds, decisions, leader, metrics, impact = load()
store = st.sidebar.selectbox("Store", ["All"] + sorted(preds.store_id.unique().tolist()))
product = st.sidebar.selectbox("Product", ["All"] + sorted(preds.product_id.unique().tolist()))
filtered = preds.copy()
if store != "All":
    filtered = filtered[filtered.store_id == store]
if product != "All":
    filtered = filtered[filtered.product_id == product]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Holdout WAPE", f"{metrics['wape']:.1%}")
c2.metric("Holdout MAE", f"{metrics['mae']:.2f} units")
c3.metric("Selected model", metrics["selected_model"])
c4.metric("Simulated inventory cost change", f"{impact['simulated_cost_reduction_pct']:+.1f}% reduction")
st.subheader("Actual vs predicted demand")
series = filtered.groupby("date", as_index=False)[["latent_demand", "prediction"]].sum()
long = series.melt(id_vars="date", var_name="Series", value_name="Units")
st.plotly_chart(px.line(long, x="date", y="Units", color="Series", title="Daily demand (aggregated selection)"), use_container_width=True)
left, right = st.columns(2)
with left:
    st.subheader("Model validation")
    st.dataframe(leader, use_container_width=True, hide_index=True)
with right:
    st.subheader("Inventory scenario")
    st.dataframe(pd.DataFrame([impact]).T.rename(columns={0:"Value"}), use_container_width=True)
st.subheader("Store-level inventory comparison")
subset = decisions.copy()
if store != "All":
    subset = subset[subset.store_id == store]
if product != "All":
    subset = subset[subset.product_id == product]
by_store = subset.groupby("store_id", as_index=False)[["ml_cost", "baseline_cost"]].sum()
st.plotly_chart(px.bar(by_store.melt(id_vars="store_id", var_name="Decision rule", value_name="Hypothetical cost"),
                       x="store_id", y="Hypothetical cost", color="Decision rule", barmode="group"), use_container_width=True)
st.info("Model forecasts use prior observed latent demand from a simulation. This is a one-step-ahead backtest, not a recursive 28-day production forecast. Cost parameters and uncertainty are illustrative; actual business impact is unverified.")
