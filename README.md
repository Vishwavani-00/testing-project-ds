# Time Series Forecasting for Retail Sales Demand

End-to-end forecasting pipeline: clean historical sales data → EDA → train multiple models → select best → generate a 30-day forward forecast → deliver an HTML report.

---

## Pipeline

```
Raw Data (CSV or synthetic)
      │
  [1] Load data
      │
  [2] Clean          fill gaps · cap outliers · daily frequency
      │
  [3] Feature Eng.   lag(1,7,30) · rolling mean(7,30) · time features
      │
  [4] EDA            trend · seasonality · decomposition · ADF test
      │
  [5] Train/Test     last 30 days held out as test
      │
  [6] Models         Naive · MovingAvg · ARIMA · RandomForest · XGBoost
      │              MAE · RMSE · MAPE → best model selected
      │
  [7] Forecast       30-day recursive prediction from best model
      │
  [8] Export         forecast CSV  +  HTML report
```

---

## Project Structure

```
├── run_pipeline.py           # Entry point — runs all steps
├── src/
│   ├── data_loader.py        # CSV loader + synthetic data generator
│   ├── preprocessing.py      # Cleaning + feature engineering
│   ├── eda.py                # EDA charts + ADF stationarity test
│   ├── models.py             # All forecasting models
│   ├── evaluation.py         # MAE, RMSE, MAPE + visualisations
│   ├── forecast.py           # 30-day forecast generation + CSV export
│   └── report.py             # HTML report generator
├── tests/
│   ├── test_preprocessing.py
│   └── test_models.py
├── data/                     # Place superstore.csv here
├── outputs/
│   ├── forecasts/            # 30-day forecast CSVs
│   └── reports/              # HTML report
├── requirements.txt
└── ds_project_requirement.md
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with synthetic data (no CSV required)
python run_pipeline.py

# Run with your Kaggle CSV
python run_pipeline.py --data data/superstore.csv

# Filter by store and product
python run_pipeline.py --data data/superstore.csv --store S1 --product P1
```

Open `outputs/reports/sales_forecast_report.html` in any browser.

---

## Models

| Model | Type | Notes |
|---|---|---|
| Naive | Baseline | Repeats last value |
| MovingAverage | Baseline | Mean of last 7 days |
| ARIMA | Statistical | Auto order selection (AIC) |
| RandomForest | ML | 100 estimators, lag features |
| XGBoost | ML | 200 estimators, lag features |

Best model selected by lowest RMSE on 30-day test set.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error (primary selection metric) |
| MAPE | Mean Absolute Percentage Error |

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Non-Functional Targets

| Requirement | Target |
|---|---|
| Runtime | < 15 minutes |
| Reproducibility | RANDOM_SEED = 42 throughout |
| Scalability | Filter by store_id / product_id |

---

## Dataset

**Source:** Kaggle — Store Sales / Retail Sales Dataset

**Required columns:** `date`, `sales`

**Optional:** `store_id`, `product_id`, `promotions`

If no CSV is provided the pipeline generates 2 years of realistic synthetic data automatically.
