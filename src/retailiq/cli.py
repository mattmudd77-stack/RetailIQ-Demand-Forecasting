"""Command-line pipeline: synthetic data -> SQLite -> ML -> inventory evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from .synthetic import generate_data
from .database import build_database, monthly_sales
from .model import run_training
from .features import make_features
from .inventory import evaluate_inventory, summarize_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="RetailIQ reproducible forecasting pipeline")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    parser.add_argument("--days", type=int, default=420)
    parser.add_argument("--stores", type=int, default=8)
    parser.add_argument("--products", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = generate_data(args.days, args.stores, args.products, args.seed)
    raw.to_csv(out / "synthetic_sales.csv", index=False)
    build_database(raw, out / "retail.db")
    monthly_sales(out / "retail.db").to_csv(out / "monthly_sales.csv", index=False)
    score = run_training(raw, out, seed=args.seed)
    # Match stored test predictions to feature rows without recalculating labels from predictions.
    feature_rows = make_features(raw)
    pred = pd.read_csv(out / "test_predictions.csv", parse_dates=["date"])
    joined = feature_rows.merge(pred[["date", "store_id", "product_id", "prediction"]],
                                on=["date", "store_id", "product_id"], how="inner", validate="one_to_one")
    evaluated = evaluate_inventory(joined)
    evaluated[["date", "store_id", "product_id", "latent_demand", "prediction", "ml_stock", "baseline_stock",
               "ml_cost", "baseline_cost", "ml_stockout_units", "baseline_stockout_units"]].to_csv(out / "inventory_decisions.csv", index=False)
    impact = summarize_inventory(evaluated)
    (out / "inventory_summary.json").write_text(json.dumps(impact, indent=2))
    print(json.dumps({"forecast": score, "inventory": impact}, indent=2))


if __name__ == "__main__":
    main()
