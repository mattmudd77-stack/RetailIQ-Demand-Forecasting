# RetailIQ — Retail Demand Forecasting & Inventory Optimization

**Forecast daily store–product demand and turn those predictions into inventory decisions.** RetailIQ combines a leakage-aware forecasting pipeline, temporal backtesting, SQL processing, and an interactive Streamlit dashboard. It compares machine-learning forecasts against a seasonal-naive baseline, then evaluates an illustrative cost-sensitive replenishment policy.

**Data:** Deterministic synthetic retail demand; no proprietary or employer data is included. All financial outcomes are hypothetical simulations, not real savings.

## Results at a glance

The following results are from the checked-in seeded **synthetic** dataset. Model selection uses two expanding-window validation folds; the final holdout is later in time.

| Forecast model | Mean validation WAPE ↓ |
| --- | ---: |
| Seasonal naive (same weekday last week, `lag_7`) | **28.71%** |
| Random Forest | 18.70% |
| Histogram Gradient Boosting | 18.59% |
| **LightGBM (selected)** | **18.47%** |

**Later holdout — selected LightGBM:** **17.43% WAPE**, **4.15 MAE**, **5.32 RMSE**, and **+0.64 mean bias**, across **5,664 store–SKU–day rows** (December 27, 2024–February 23, 2025). The seasonal-naive figure above is a *validation* score, **not** a measured baseline holdout score; don't interpret the difference as a holdout improvement.

### Illustrative inventory simulation

| Policy | Hypothetical total cost | Stockout units |
| --- | ---: | ---: |
| Seasonal-naive inventory baseline | $260,493 | 16,339 |
| Model-based newsvendor policy | $127,911 | 2,636 |
| **Difference** | **−$132,582 (−50.9%)** | **−13,703** |

These are **simulated scenario outcomes**, not realized retailer savings. The policies are evaluated against the same synthetic demand, with simplified inventory assumptions and heuristic (uncalibrated) uncertainty.

![Forecast model validation comparison](docs/validation-wape.svg)

![Illustrative inventory cost comparison](docs/inventory-cost.svg)

**Try it locally:** `pip install -e '.[dev,boost]'` → `retailiq --out artifacts` → `streamlit run dashboard.py`. No public hosted dashboard is currently claimed.

## Business problem

A multi-store retailer needs to estimate daily demand at the store–SKU level and make replenishment decisions. Forecasting demand alone is not enough: understocking and overstocking have different costs. RetailIQ compares several forecast models and evaluates an illustrative inventory decision rule against a seasonal-naive baseline.

## Architecture

```text
Deterministic simulator → CSV + SQLite (SQL aggregation)
                      ↓
Point-in-time features (lags, rolling windows, calendar, pricing)
                      ↓
Expanding-window validation → model selection → final temporal holdout
                      ↓
One-step-ahead predictions → newsvendor inventory simulation
                      ↓
Saved artifacts → interactive Streamlit dashboard
                      ↓
pytest + GitHub Actions CI + Docker deployment configuration
```

## Model design

- **Targets:** `latent_demand`, which is visible only because the dataset is simulated. In real retail data, sales are often censored by stockouts; this target would require a defensible demand-estimation strategy.
- **Features:** 1/7/14/28-day demand lags, 7/28-day trailing means and standard deviations, store and product IDs, price, promotion, holiday, and calendar attributes. All demand-derived features use **prior dates only**.
- **Models:** seasonal naive (`lag_7`), random forest, histogram gradient boosting, and **optional** LightGBM.
- **Validation:** two expanding-window validation folds, select lowest mean WAPE, retrain on all pre-test history, evaluate on a strictly later test period. No random splitting of time-series rows.
- **Metrics:** WAPE, MAE, RMSE, mean forecast bias. Validation metrics and final holdout metrics are saved separately.
- **Forecasting scope:** the holdout is a **one-step-ahead historical backtest** that uses actual previous-day demand to form lag features. It is **not** a recursive multi-step deployment forecast, and future covariate availability is not simulated.

## Inventory policy

For an estimated demand mean μ and heuristic standard deviation σ, the order-up-to level is

`Q = max(0, ceil(μ + Φ⁻¹(Cu / (Cu + Co)) · σ))`

where `Cu` is the unit underage penalty and `Co` the unit overage penalty. Defaults: **$12** and **$3** per unit, respectively. Uncertainty is an **assumed heuristic** `max(2, sqrt(predicted demand))`, not a calibrated prediction interval. The baseline orders the prior week's same-day demand (`lag_7`). Both policies are evaluated on the **same synthetic realized demand**. This comparison excludes lead times, minimum order quantities, fixed ordering costs, service-level guarantees, and supply-chain constraints.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev,boost]'
retailiq --out artifacts
pytest -q
streamlit run dashboard.py
```

LightGBM is optional; the core pipeline works without it. To run without LightGBM, install `pip install -e '.[dev]'`. The checked-in `artifacts/` metrics are generated on the default seeded synthetic dataset.

## Docker

```bash
docker build -t retailiq .
docker run --rm -p 8501:8501 retailiq
```

Open `http://localhost:8501`.

## Repository layout

```text
src/retailiq/       synthetic generation, SQL, features, training, inventory, CLI
artifacts/          example metrics, predictions, inventory comparisons (generated)
tests/              reproducibility, leakage, SQL, inventory and pipeline tests
dashboard.py        Streamlit visualizations and filters
pyproject.toml      installable package + dependencies
.github/workflows/  CI testing
Dockerfile          reproducible dashboard container
```

## What these results do—and do not—show

- **Why WAPE?** It aggregates absolute forecast errors relative to aggregate demand, avoiding the division-by-zero instability of per-row MAPE on low or zero demand. It still emphasizes high-volume series.
- **Why temporal validation?** Random splits would let later observations influence model selection for earlier dates. Every demand-derived lag or rolling feature uses observations strictly before the forecast date.
- **Why the critical ratio?** In the single-period newsvendor model, the optimal demand quantile balances the marginal underage penalty against the marginal overage penalty: `Cu / (Cu + Co)`.
- **Why heuristic σ?** `max(2, sqrt(predicted demand))` is a deliberately simple scale assumption for a demonstration; it is **not** calibrated to residuals or a prediction interval. Consequently, inventory cost comparisons should not be treated as validated operational policy improvements.
- **What is not evaluated?** Recursive forecasts, stockout-censored observed sales, lead times, real-world demand uncertainty, and deployment drift.

## Technical limitations and next steps

1. Replace synthetic data with properly licensed public retail data and handle censored sales.
2. Implement **recursive multi-horizon forecasting** with known-in-advance covariates, and rolling-origin evaluation at the requested horizon.
3. Add quantile or conformal prediction intervals and calibrate uncertainty before inventory optimization.
4. Incorporate lead times, supplier constraints, lost-sales dynamics, and a true sequential inventory simulation.
5. Add MLflow experiment tracking, drift monitoring, scheduled retraining, and cloud deployment.

**Ethics:** No customer-level personally identifiable information is generated or processed. This is a learning/demo system, not a production retail planning tool.