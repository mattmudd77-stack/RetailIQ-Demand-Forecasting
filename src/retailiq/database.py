"""SQLite ingestion and SQL aggregation with uniqueness constraints."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd


def build_database(data: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute("DROP TABLE IF EXISTS daily_sales")
        con.execute("""CREATE TABLE daily_sales (
          date TEXT NOT NULL, store_id TEXT NOT NULL, product_id TEXT NOT NULL,
          price REAL NOT NULL, promotion INTEGER NOT NULL, holiday INTEGER NOT NULL,
          inventory_available INTEGER NOT NULL, units_sold INTEGER NOT NULL,
          latent_demand INTEGER NOT NULL, stockout INTEGER NOT NULL,
          PRIMARY KEY(date, store_id, product_id))""")
        records = data.assign(date=pd.to_datetime(data.date).dt.strftime("%Y-%m-%d"))
        cols = ["date", "store_id", "product_id", "price", "promotion", "holiday",
                "inventory_available", "units_sold", "latent_demand", "stockout"]
        records[cols].to_sql("daily_sales", con, if_exists="append", index=False)


def monthly_sales(path: str | Path) -> pd.DataFrame:
    with sqlite3.connect(path) as con:
        return pd.read_sql_query("""SELECT substr(date,1,7) AS month, store_id,
          SUM(units_sold) AS units_sold, ROUND(SUM(units_sold * price),2) AS revenue,
          SUM(stockout) AS stockout_days FROM daily_sales
          GROUP BY month, store_id ORDER BY month, store_id""", con)
