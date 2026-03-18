"""
run_pipeline.py
===============
End-to-end Time Series Forecasting Pipeline for Retail Sales Demand.

Usage:
    python run_pipeline.py                              # synthetic data
    python run_pipeline.py --data data/superstore.csv  # your CSV
    python run_pipeline.py --data data/superstore.csv --store S1 --product P1

Pipeline Steps:
    1. Load / generate data
    2. Clean (fill gaps, cap outliers, enforce daily frequency)
    3. Engineer features (lag, rolling, time-based)
    4. Exploratory Data Analysis
    5. Train/test split (last 30 days = test)
    6. Train all models + evaluate (MAE, RMSE, MAPE)
    7. Generate 30-day forecast using best model
    8. Export forecast CSV
    9. Generate HTML report
"""
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader   import load_data, generate_sample_data
from src.preprocessing import clean, engineer_features, train_test_split_ts
from src.eda           import run_eda
from src.models        import (NaiveForecaster, MovingAverageForecaster,
                               ARIMAForecaster, LinearRegressionForecaster,
                               RandomForestForecaster, XGBoostForecaster)
from src.evaluation    import evaluate_all, plot_predictions, \
                              plot_metrics_comparison, plot_residuals
from src.forecast      import generate_forecast, export_forecast, plot_forecast
from src.report        import generate_html_report

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FORECAST_DIR = os.path.join(BASE_DIR, "outputs", "forecasts")
REPORT_DIR   = os.path.join(BASE_DIR, "outputs", "reports")


def run(data_path: str = None,
        store_id:  str = None,
        product_id: str = None):

    print("\n" + "="*62)
    print("  Time Series Forecasting — Retail Sales Demand")
    print("="*62)

    # ── 1. Load data ────────────────────────────────────────────
    print("\n▶  Step 1: Loading data ...")
    if data_path and os.path.exists(data_path):
        df = load_data(data_path)
        if store_id and "store_id" in df.columns:
            df = df[df["store_id"] == store_id]
        if product_id and "product_id" in df.columns:
            df = df[df["product_id"] == product_id]
        if len(df) == 0:
            raise ValueError("No rows after filtering — check store/product IDs.")
    else:
        print("  No CSV provided — using 2-year synthetic data.")
        df = generate_sample_data(n_days=730, seed=RANDOM_SEED)

    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "store_id", "product_id", "sales", "promotions"]].copy()

    # ── 2. Clean ────────────────────────────────────────────────
    print("\n▶  Step 2: Cleaning ...")
    df_clean = clean(df)

    # ── 3. Feature engineering ──────────────────────────────────
    print("\n▶  Step 3: Feature engineering ...")
    df_feat = engineer_features(df_clean)

    # ── 4. EDA ──────────────────────────────────────────────────
    print("\n▶  Step 4: EDA ...")
    eda_results = run_eda(df_clean)

    # ── 5. Train / test split ───────────────────────────────────
    print("\n▶  Step 5: Train/test split (last 30 days = test) ...")
    train_df, test_df = train_test_split_ts(df_feat, test_days=30)
    train_series = train_df.set_index("date")["sales"]
    test_series  = test_df.set_index("date")["sales"]
    n_test       = len(test_df)

    # ── 6. Train models + evaluate ──────────────────────────────
    print("\n▶  Step 6: Training & evaluating models ...")
    eval_results = {}

    print("  [1/5] Naive ...")
    naive = NaiveForecaster().fit(train_series)
    eval_results["Naive"] = {
        "actual": test_series.values, "predicted": naive.predict(n_test)}

    print("  [2/5] Moving Average ...")
    ma = MovingAverageForecaster(window=7).fit(train_series)
    eval_results["MovingAverage"] = {
        "actual": test_series.values, "predicted": ma.predict(n_test)}

    print("  [3/5] ARIMA (auto order) ...")
    arima_model = None
    try:
        arima_model = ARIMAForecaster().fit(train_series)
        eval_results["ARIMA"] = {
            "actual": test_series.values,
            "predicted": arima_model.predict(n_test)}
    except Exception as e:
        print(f"  ARIMA skipped: {e}")

    print("  [4/5] Random Forest ...")
    rf = RandomForestForecaster(n_estimators=100)
    rf.fit(train_df, sales_col="sales")
    eval_results["RandomForest"] = {
        "actual": test_series.values, "predicted": rf.predict(n_test)}

    print("  [5/5] XGBoost ...")
    xgb = XGBoostForecaster()
    xgb.fit(train_df, sales_col="sales")
    eval_results["XGBoost"] = {
        "actual": test_series.values, "predicted": xgb.predict(n_test)}

    metrics_df      = evaluate_all(eval_results)
    best_model_name = metrics_df.iloc[0]["Model"]
    print(f"\n  Best model: {best_model_name} "
          f"(RMSE: {metrics_df.iloc[0]['RMSE']})")
    print(metrics_df.to_string(index=False))

    eval_figs = {
        "predictions_fig": plot_predictions(eval_results, test_df["date"]),
        "metrics_fig":     plot_metrics_comparison(metrics_df),
        "residuals_fig":   plot_residuals(eval_results, best_model_name),
    }

    # ── 7. Generate 30-day forecast ─────────────────────────────
    print("\n▶  Step 7: Generating 30-day forecast ...")
    model_map = {
        "Naive":           naive,
        "MovingAverage":   ma,
        "ARIMA":           arima_model if arima_model else ma,
        "RandomForest":    rf,
        "XGBoost":         xgb,
    }
    best_model = model_map[best_model_name]

    # Refit ML model on full dataset for final forecast
    if hasattr(best_model, "_feature_cols"):
        best_model.fit(df_feat, sales_col="sales")

    forecast_df = generate_forecast(best_model, df_feat, horizon=30)

    # ── 8. Export forecast CSV ──────────────────────────────────
    print("\n▶  Step 8: Exporting forecast CSV ...")
    export_forecast(forecast_df, FORECAST_DIR, best_model_name)

    # ── 9. Generate HTML report ─────────────────────────────────
    print("\n▶  Step 9: Generating HTML report ...")
    forecast_fig  = plot_forecast(df_clean, forecast_df, best_model_name)
    report_path   = os.path.join(REPORT_DIR, "sales_forecast_report.html")
    generate_html_report(
        eda_results     = eda_results,
        metrics_df      = metrics_df,
        best_model_name = best_model_name,
        forecast_df     = forecast_df,
        eval_figs       = eval_figs,
        forecast_fig    = forecast_fig,
        output_path     = report_path,
    )

    print("\n" + "="*62)
    print("  ✅  Pipeline Complete!")
    print(f"  📊  Report  : {report_path}")
    print(f"  📁  Forecast: {FORECAST_DIR}/")
    print("="*62 + "\n")
    return report_path, forecast_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Time Series Forecasting Pipeline — Retail Sales Demand")
    parser.add_argument("--data",    type=str, default=None,
                        help="Path to sales CSV (optional; synthetic used if absent)")
    parser.add_argument("--store",   type=str, default=None,
                        help="Filter by store_id")
    parser.add_argument("--product", type=str, default=None,
                        help="Filter by product_id")
    args = parser.parse_args()
    run(data_path=args.data, store_id=args.store, product_id=args.product)
