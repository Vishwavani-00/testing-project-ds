# Time Series Forecasting for Retail Sales Demand

An end-to-end machine learning pipeline that forecasts retail sales demand using historical time series data — enabling better inventory planning and decision-making.

---

## Objective

Predict future daily sales for each store/product over a 30-day horizon using baseline, statistical, and machine learning models.

---

## Pipeline

```
Raw Data (CSV or Synthetic)
       │
  [1] Load data
  [2] Clean          — fill gaps, cap outliers, ensure daily frequency
  [3] Feature Eng.   — lag(1,7,30), rolling mean(7,30), time-based features
  [4] EDA            — trend, seasonality, decomposition, ADF stationarity
  [5] Train/Test     — last 30 days held out as test
  [6] Models         — Naive · MovingAvg · ARIMA · RandomForest · XGBoost
                       MAE · RMSE · MAPE → best model auto-selected
  [7] Forecast       — 30-day recursive prediction from best model
  [8] Export         — forecast CSV + HTML report
```

---

## Project Structure

```
├── run_pipeline.py              # Entry point — runs all 8 steps end-to-end
├── src/
│   ├── data_loader.py           # CSV loader + synthetic data generator
│   ├── preprocessing.py         # Cleaning + feature engineering
│   ├── eda.py                   # EDA charts + ADF stationarity test
│   ├── models.py                # All forecasting models
│   ├── evaluation.py            # MAE, RMSE, MAPE + visualizations
│   ├── forecast.py              # 30-day forecast generation + CSV export
│   └── report.py                # Self-contained HTML report generator
├── tests/
│   ├── test_preprocessing.py    # Unit tests for preprocessing
│   └── test_models.py           # Unit tests for models
├── data/                        # Place superstore.csv here (optional)
├── outputs/
│   ├── forecasts/               # Generated forecast CSVs
│   └── reports/                 # Generated HTML reports
├── requirements.txt
└── ds_project_requirement.md    # Original PRD
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with synthetic data (no CSV needed)
python run_pipeline.py

# 3. Run with your own Kaggle CSV
python run_pipeline.py --data data/superstore.csv

# 4. Filter by store and product
python run_pipeline.py --data data/superstore.csv --store S1 --product P1
```

Open `outputs/reports/sales_forecast_report.html` in any browser to view the full report.

---

## Models

| Model | Type | Notes |
|---|---|---|
| Naive | Baseline | Repeats last observed value |
| MovingAverage | Baseline | Mean of last 7 days |
| ARIMA | Statistical | Auto order selection by AIC |
| RandomForest | ML | 100 estimators, lag + rolling features |
| XGBoost | ML | 200 estimators, lag + rolling features |

Best model is auto-selected by lowest RMSE on the 30-day test set.

---

## Results (Synthetic Data)

| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| 🏆 XGBoost | 13.73 | 16.66 | 8.44 |
| RandomForest | 14.32 | 17.04 | 8.87 |
| ARIMA | 17.06 | 20.26 | 10.51 |
| MovingAverage | 17.40 | 20.82 | 11.48 |
| Naive | 18.50 | 22.54 | 10.95 |

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Statistical Models | statsmodels (ARIMA) |
| ML Models | scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Testing | pytest |

---

## Dataset

**Source:** Kaggle — Store Sales / Retail Sales Dataset

**Required columns:** `date`, `sales`

**Optional:** `store_id`, `product_id`, `promotions`

If no CSV is provided, the pipeline auto-generates 2 years of realistic synthetic data (trend + weekly seasonality + noise + promotions).
