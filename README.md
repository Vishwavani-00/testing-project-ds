# Time Series Forecasting for Retail Sales Demand

A complete, production-ready forecasting pipeline that ingests historical retail sales data, performs EDA, trains multiple models, selects the best one, and generates a 30-day forward forecast with an HTML report.

---

## Pipeline Architecture

```
Raw Data (CSV or Synthetic)
        │
        ▼
  [1] Data Loading          — load_data() / generate_sample_data()
        │
        ▼
  [2] Cleaning              — fill gaps, interpolate, cap outliers
        │
        ▼
  [3] Feature Engineering   — lag, rolling, time-based features
        │
        ▼
  [4] EDA                   — trend, seasonality, ADF stationarity
        │
        ▼
  [5] Model Training        — Naive, MA, ARIMA, RF, XGBoost
        │
        ▼
  [6] Evaluation            — MAE, RMSE, MAPE → best model selected
        │
        ▼
  [7] 30-Day Forecast       — forward predictions from best model
        │
        ▼
  [8] Export & Report       — forecast CSV + HTML report
```

---

## Project Structure

```
├── run_pipeline.py             # Main entry point — runs full pipeline
├── src/
│   ├── data_loader.py          # Load CSV or generate synthetic data
│   ├── preprocessing.py        # Cleaning + feature engineering
│   ├── eda.py                  # EDA charts + ADF test
│   ├── models.py               # All forecasting models
│   ├── evaluation.py           # MAE, RMSE, MAPE + comparison plots
│   ├── forecast.py             # 30-day forecast generation + export
│   └── report.py               # HTML report generator
├── tests/
│   ├── test_preprocessing.py   # Unit tests for preprocessing
│   └── test_models.py          # Unit tests for models
├── data/                       # Place your superstore.csv here
├── outputs/
│   ├── forecasts/              # Forecast CSVs exported here
│   └── reports/                # HTML report saved here
├── requirements.txt
└── ds_project_requirement.md
```

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| Naive | Baseline | Repeats last observed value |
| MovingAverage | Baseline | Mean of last 7 days |
| ARIMA | Statistical | Auto order selection (best AIC) |
| RandomForest | ML | 100 trees, lag + rolling features |
| XGBoost | ML | Gradient boosting, lag + rolling features |

**Best model** is auto-selected by lowest RMSE on the 30-day test set.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with Synthetic Data (no CSV needed)

```bash
python run_pipeline.py
```

### 3. Run with Your Own Data

Place your Kaggle Store Sales CSV in `data/` then:

```bash
python run_pipeline.py --data data/superstore.csv
python run_pipeline.py --data data/superstore.csv --store S1 --product P1
```

### 4. View Report

Open `outputs/reports/sales_forecast_report.html` in any browser.

### 5. Run Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Output

| Output | Location |
|--------|----------|
| HTML Report | `outputs/reports/sales_forecast_report.html` |
| Forecast CSV | `outputs/forecasts/forecast_<model>_30day.csv` |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error — average absolute deviation |
| RMSE | Root Mean Squared Error — penalises large errors |
| MAPE | Mean Absolute Percentage Error — % accuracy |

---

## Non-Functional Targets

| Requirement | Target |
|-------------|--------|
| Pipeline Runtime | < 15 minutes |
| Reproducibility | `RANDOM_SEED = 42` throughout |
| Scalability | Filter by store_id / product_id |
| Data Freshness | Re-run with new CSV anytime |

---

## Dataset

**Source:** Kaggle — [Store Sales / Retail Sales Dataset](https://www.kaggle.com/datasets)

**Required columns:** `date`, `sales`

**Optional:** `store_id`, `product_id`, `promotions`

If no CSV is provided, the pipeline generates 2 years of realistic synthetic data.
