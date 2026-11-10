"""Temporal validation, reproducible model selection, and untouched holdout scoring."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .features import FEATURES, assert_valid_history, make_features


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(0, np.asarray(predicted, dtype=float))
    return {"mae": round(float(mean_absolute_error(actual, predicted)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
            "wape": round(float(np.abs(actual - predicted).sum() / max(np.abs(actual).sum(), 1)), 4),
            "bias": round(float((predicted - actual).mean()), 4)}


def candidates(seed: int = 42) -> dict:
    models = {
        "seasonal_naive": None,
        "random_forest": RandomForestRegressor(n_estimators=65, max_depth=13, min_samples_leaf=3,
                                                random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=25,
                                                                 learning_rate=.07, l2_regularization=1., random_state=seed),
    }
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm"] = LGBMRegressor(n_estimators=180, learning_rate=.045, num_leaves=24,
                                           verbosity=-1, random_state=seed, n_jobs=2)
    except ImportError:
        pass
    return models


def predict(model, frame: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.maximum(0, frame.lag_7.to_numpy(dtype=float))
    return np.maximum(0, model.predict(frame[FEATURES]))


def run_training(raw: pd.DataFrame, outdir: str | Path, seed: int = 42, target: str = "latent_demand") -> dict:
    """Two rolling-origin validation folds, then a strictly later test interval.

    Evaluation uses historical observed demand for lags (one-step-ahead backtest).
    This is NOT a recursive multi-day deployment forecast.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = make_features(raw, target=target)
    assert_valid_history(data)
    dates = np.sort(data.date.unique())
    if len(dates) < 75:
        raise ValueError("Not enough history after lag warmup")
    cut1, cut2, cut3 = (dates[int(len(dates) * p)] for p in (.55, .70, .85))
    folds = [(data.date < cut1, (data.date >= cut1) & (data.date < cut2)),
             (data.date < cut2, (data.date >= cut2) & (data.date < cut3))]
    test = data[data.date >= cut3].copy()
    history = data[data.date < cut3].copy()
    results = []
    for name in candidates(seed):
        fold_scores = []
        for train_mask, val_mask in folds:
            train, val = data.loc[train_mask], data.loc[val_mask]
            model = candidates(seed)[name]
            if model is not None:
                model.fit(train[FEATURES], train[target])
            fold_scores.append(metrics(val[target], predict(model, val))["wape"])
        results.append({"model": name, "validation_wape": round(float(np.mean(fold_scores)), 4),
                        "fold_1_wape": fold_scores[0], "fold_2_wape": fold_scores[1]})
    leaderboard = pd.DataFrame(results).sort_values("validation_wape").reset_index(drop=True)
    winner = leaderboard.iloc[0]["model"]
    fitted = candidates(seed)[winner]
    if fitted is not None:
        fitted.fit(history[FEATURES], history[target])
    test["prediction"] = predict(fitted, test)
    score = metrics(test[target], test.prediction)
    score["seasonal_naive_holdout"] = metrics(test[target], predict(None, test))
    if "expected_demand" in test:
        score["oracle_expected_demand_holdout"] = metrics(test[target], test["expected_demand"])
    score.update({"selected_model": winner, "target": target, "validation_scheme": "two expanding folds + later holdout",
                  "forecast_scope": "one-step-ahead backtest with observed prior demand", "test_rows": len(test),
                  "test_start": str(pd.Timestamp(cut3).date()), "test_end": str(test.date.max().date())})
    leaderboard.to_csv(outdir / "leaderboard.csv", index=False)
    test[["date", "store_id", "product_id", target, "prediction", "price", "promotion", "inventory_available", "units_sold"]].to_csv(outdir / "test_predictions.csv", index=False)
    with (outdir / "metrics.json").open("w") as f:
        json.dump(score, f, indent=2)
    joblib.dump({"model": fitted, "name": winner, "features": FEATURES, "target": target}, outdir / "model.joblib")
    return score
