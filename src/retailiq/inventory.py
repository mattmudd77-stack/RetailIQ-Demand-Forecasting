"""Newsvendor inventory decisions with honest simulated cost assumptions."""
from __future__ import annotations
from statistics import NormalDist
import numpy as np
import pandas as pd


def optimal_stock(predicted_demand: float, demand_std: float, underage_cost: float = 12., overage_cost: float = 3.) -> int:
    """Normal approximation to the critical-fractile newsvendor order-up-to level."""
    if min(predicted_demand, demand_std, underage_cost, overage_cost) < 0:
        raise ValueError("Demand, uncertainty, and costs must be nonnegative")
    if underage_cost + overage_cost <= 0:
        raise ValueError("At least one cost must be positive")
    fractile = np.clip(underage_cost / (underage_cost + overage_cost), .0001, .9999)
    return max(0, int(np.ceil(predicted_demand + NormalDist().inv_cdf(float(fractile)) * demand_std)))


def evaluate_inventory(test: pd.DataFrame, target: str = "latent_demand", underage_cost: float = 12.,
                       overage_cost: float = 3.) -> pd.DataFrame:
    """Paired decision comparison on same realized synthetic latent demand.

    Baseline: order yesterday's same-weekday demand. Proposed: predicted demand
    plus safety stock from an assumed uncertainty estimate; no lead-time simulation.
    """
    if target not in test or "prediction" not in test or "lag_7" not in test:
        raise ValueError("Need actual demand, prediction and lag_7")
    frame = test.copy()
    # Conservative heuristic; uncertainty is a scenario parameter, not calibrated predictive intervals.
    sigma = np.maximum(2., np.sqrt(np.maximum(frame.prediction.to_numpy(dtype=float), 0)))
    frame["ml_stock"] = [optimal_stock(mu, sd, underage_cost, overage_cost)
                         for mu, sd in zip(frame.prediction, sigma)]
    frame["baseline_stock"] = np.maximum(0, np.ceil(frame.lag_7)).astype(int)
    for prefix in ("ml", "baseline"):
        stock = frame[f"{prefix}_stock"].to_numpy()
        actual = frame[target].to_numpy()
        frame[f"{prefix}_stockout_units"] = np.maximum(actual - stock, 0)
        frame[f"{prefix}_excess_units"] = np.maximum(stock - actual, 0)
        frame[f"{prefix}_cost"] = underage_cost * frame[f"{prefix}_stockout_units"] + overage_cost * frame[f"{prefix}_excess_units"]
    return frame


def summarize_inventory(frame: pd.DataFrame) -> dict:
    baseline = float(frame.baseline_cost.sum())
    proposed = float(frame.ml_cost.sum())
    return {"baseline_cost": round(baseline, 2), "ml_cost": round(proposed, 2),
            "simulated_cost_reduction_pct": round(100 * (baseline - proposed) / baseline, 2) if baseline else 0.,
            "baseline_stockout_units": int(frame.baseline_stockout_units.sum()),
            "ml_stockout_units": int(frame.ml_stockout_units.sum()),
            "note": "Hypothetical costs on synthetic data; not observed financial savings."}
