"""Deterministic simulated retail demand with explicit stockout censoring."""
from __future__ import annotations
import numpy as np
import pandas as pd


def generate_data(days: int = 420, stores: int = 8, products: int = 12, seed: int = 42) -> pd.DataFrame:
    if days < 100 or stores < 1 or products < 1:
        raise ValueError("Use at least 100 days and positive store/product counts")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    store_effect = rng.uniform(0.65, 1.5, stores)
    product_effect = rng.uniform(5, 22, products)
    product_price = rng.uniform(12, 90, products)
    for s in range(stores):
        for p in range(products):
            price = float(product_price[p])
            level = product_effect[p] * store_effect[s]
            for t, date in enumerate(dates):
                weekend = int(date.dayofweek >= 5)
                promotion = int(rng.random() < 0.13)
                holiday = int((date.month == 12 and date.day >= 15) or (date.month == 11 and date.day >= 22))
                seasonality = 1 + 0.20 * np.sin(2 * np.pi * t / 365 + p / 3)
                trend = 1 + 0.0005 * t
                expected = level * seasonality * trend * (1 + .16 * weekend + .48 * promotion + .35 * holiday)
                expected *= (price / 45) ** (-.15)
                demand = int(rng.poisson(max(expected, .1)))
                # A subset of days are supply constrained; demand is retained as simulation ground truth.
                inventory_available = max(0, int(round(expected * rng.uniform(.35, .8)))) if rng.random() < .09 else max(demand, int(expected * 1.5))
                units_sold = min(demand, inventory_available)
                rows.append((date, f"S{s+1:02d}", f"P{p+1:02d}", round(price * (.90 if promotion else 1), 2),
                             promotion, holiday, weekend, inventory_available, units_sold, demand,
                             int(units_sold < demand)))
    return pd.DataFrame(rows, columns=["date", "store_id", "product_id", "price", "promotion", "holiday",
                                        "weekend", "inventory_available", "units_sold", "latent_demand", "stockout"])
