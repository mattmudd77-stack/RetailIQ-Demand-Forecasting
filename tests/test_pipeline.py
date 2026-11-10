import numpy as np
import pandas as pd
import pytest
from retailiq.synthetic import generate_data
from retailiq.features import make_features
from retailiq.inventory import optimal_stock, evaluate_inventory, summarize_inventory
from retailiq.model import metrics, run_training
from retailiq.database import build_database, monthly_sales


def test_reproducible_data():
    a = generate_data(110, 2, 2, 7)
    pd.testing.assert_frame_equal(a, generate_data(110, 2, 2, 7))
    assert (a.units_sold <= a.latent_demand).all()
    assert len(a) == 440


def test_no_future_leakage():
    df = generate_data(110, 1, 1)
    base = make_features(df)
    later = df.copy()
    later.loc[later.date == later.date.max(), "latent_demand"] = 100000
    changed = make_features(later)
    pd.testing.assert_frame_equal(base.iloc[:-1], changed.iloc[:-1])
    assert base.iloc[-1].lag_1 == changed.iloc[-1].lag_1


def test_inventory_monotonic():
    assert optimal_stock(20, 5, 12, 3) >= optimal_stock(20, 5, 3, 12)
    assert optimal_stock(0, 0) == 0
    with pytest.raises(ValueError):
        optimal_stock(-1, 1)


def test_sql_roundtrip(tmp_path):
    df = generate_data(110, 1, 1)
    db = tmp_path / "data.db"
    build_database(df, db)
    assert monthly_sales(db).units_sold.sum() == df.units_sold.sum()


def test_metrics():
    m = metrics(np.array([0, 10]), np.array([0, 8]))
    assert m["mae"] == 1.0
    assert m["wape"] == 0.2


def test_end_to_end(tmp_path):
    df = generate_data(120, 1, 2)
    result = run_training(df, tmp_path)
    assert result["test_rows"] > 0
    preds = pd.read_csv(tmp_path / "test_predictions.csv", parse_dates=["date"])
    feat = make_features(df)
    joined = feat.merge(preds[["date", "store_id", "product_id", "prediction"]], on=["date", "store_id", "product_id"])
    impact = summarize_inventory(evaluate_inventory(joined))
    assert np.isfinite(impact["ml_cost"])
