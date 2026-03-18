# Time Series Forecasting for Retail Sales Demand

## Objective
Predict future sales demand using historical time series data to enable better inventory planning and decision-making.

## Project Structure
```
data/                         # Raw and generated datasets
src/                          # Source modules (preprocessing, EDA, models, evaluation)
notebooks/                    # Analysis notebooks/scripts
reports/                      # HTML report + EDA charts
outputs/forecasts/            # 30-day forecast CSV
run_pipeline.py               # Main pipeline runner
requirements.txt              # Python dependencies
ds_project_requirement.md     # Original PRD
```

## How to Run
```bash
pip install -r requirements.txt
python3 run_pipeline.py
```

## Technology Stack
- **Language:** Python 3.10+
- **Libraries:** pandas, numpy, statsmodels, scikit-learn, xgboost, matplotlib
- **Models:** Naive, Moving Average, ARIMA/SARIMA, XGBoost
- **Output:** HTML report, 30-day forecast CSV

## Deliverables
- `reports/retail-sales-forecasting-report-v26.07.01.html` — MuSigma-styled interactive HTML report
- `outputs/forecasts/forecast_30days.csv` — 30-day sales forecast per store/product
