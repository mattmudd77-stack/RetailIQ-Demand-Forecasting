"""Point-in-time safe features; lagged demand only uses prior days."""
from __future__ import annotations
import numpy as np
import pandas as pd

LAGS = (1, 7, 14, 28)
WINDOWS = (7, 28)
FEATURES = ["price", "promotion", "holiday", "weekend", "dow", "month", "day_of_year", "trend_day",
            "store_code", "product_code", "lag_1", "lag_7", "lag_14", "lag_28", "mean_7", "mean_28", "std_7", "std_28"]


def make_features(df: pd.DataFrame, target: str = "latent_demand") -> pd.DataFrame:
    """Lag/rolling values strictly precede the row's date within each store/product."""
    if target not in df:
        raise KeyError(target)
    data = df.copy().sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    if data.duplicated(["store_id", "product_id", "date"]).any():
        raise ValueError("Duplicate store/product/date rows")
    data["date"] = pd.to_datetime(data["date"])
    data["dow"] = data.date.dt.dayofweek
    data["month"] = data.date.dt.month
    data["day_of_year"] = data.date.dt.dayofyear
    data["trend_day"] = (data.date - data.date.min()).dt.days
    data["store_code"] = data.store_id.str.extract(r"(\d+)$")[0].astype(int)
    data["product_code"] = data.product_id.str.extract(r"(\d+)$")[0].astype(int)
    groups = data.groupby(["store_id", "product_id"], sort=False)[target]
    for lag in LAGS:
        data[f"lag_{lag}"] = groups.shift(lag)
    for window in WINDOWS:
        shifted = groups.shift(1)
        data[f"mean_{window}"] = shifted.groupby([data.store_id, data.product_id]).transform(
            lambda x: x.rolling(window, min_periods=window).mean())
        data[f"std_{window}"] = shifted.groupby([data.store_id, data.product_id]).transform(
            lambda x: x.rolling(window, min_periods=window).std())
    return data.dropna(subset=FEATURES).reset_index(drop=True)


def assert_valid_history(data: pd.DataFrame) -> None:
    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite model features")
