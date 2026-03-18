"""
run_pipeline.py
================
End-to-end Time Series Forecasting Pipeline for Retail Sales Demand.

Usage:
    python run_pipeline.py                          # Uses synthetic data
    python run_pipeline.py --data data/superstore.csv  # Uses your own CSV
    python run_pipeline.py --data data/superstore.csv --store S1 --product P1

Steps:
    1. Load / generate data
    2. Clean & feature engineer
    3. EDA
    4. Train/test split
    5. Train all models + evaluate
    6. Generate 30-day forecast with best model
    7. Export forecast CSV
    8. Generate HTML report
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data, generate_sample_data
from src.preprocessing import clean, engineer_features, train_test_split_ts
from src.eda import run_eda
from src.models import (
    NaiveForecaster, MovingAverageForecaster, ARIMAForecaster,
    LinearRegressionForecaster, RandomForestForecaster, XGBoostForecaster
)
from src.evaluation import evaluate_all, plot_predictions, plot_metrics_comparison, plot_residuals
from src.forecast import generate_forecast, export_forecast, plot_forecast
from src.report import generate_html_report

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(__file__)
OUTPUT_FORECAST_DIR = os.path.join(BASE_DIR, "outputs", "forecasts")
OUTPUT_REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")


def run(data_path: str = None, store_id: str = None, product_id: str = None):
    print("\n" + "="*60)
    print("  Time Series Forecasting — Retail Sales Demand")
    print("="*60 + "\n")

    # ── Step 1: Load Data ────────────────────────────────────────
    print("▶ Step 1: Loading data ...")
    if data_path and os.path.exists(data_path):
        df = load_data(data_path)
        # Filter to specific store/product if provided
        if store_id and "store_id" in df.columns:
            df = df[df["store_id"] == store_id]
        if product_id and "product_id" in df.columns:
            df = df[df["product_id"] == product_id]
        if len(df) == 0:
            raise ValueError("No rows after filtering. Check store_id / product_id.")
    else:
        print("  No data file provided — using synthetic data.")
        df = generate_sample_data(n_days=730)

    df = df[["date", "store_id", "product_id", "sales", "promotions"]].copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Step 2: Clean & Feature Engineering ──────────────────────
    print("\n▶ Step 2: Preprocessing ...")
    df_clean = clean(df)
    df_features = engineer_features(df_clean)

    # ── Step 3: EDA ───────────────────────────────────────────────
    print("\n▶ Step 3: EDA ...")
    eda_results = run_eda(df_clean)

    # ── Step 4: Train/Test Split ──────────────────────────────────
    print("\n▶ Step 4: Train/Test split (last 30 days as test) ...")
    train_df, test_df = train_test_split_ts(df_features, test_days=30)
    train_series = train_df.set_index("date")["sales"]
    test_series = test_df.set_index("date")["sales"]
    n_test = len(test_df)

    # ── Step 5: Train & Evaluate Models ──────────────────────────
    print("\n▶ Step 5: Training models ...")
    eval_results = {}

    # Naive
    print("  [1/5] Naive ...")
    naive = NaiveForecaster().fit(train_series)
    eval_results["Naive"] = {"actual": test_series.values, "predicted": naive.predict(n_test)}

    # Moving Average
    print("  [2/5] Moving Average ...")
    ma = MovingAverageForecaster(window=7).fit(train_series)
    eval_results["MovingAverage"] = {"actual": test_series.values, "predicted": ma.predict(n_test)}

    # ARIMA
    print("  [3/5] ARIMA (auto order selection) ...")
    try:
        arima = ARIMAForecaster().fit(train_series)
        eval_results["ARIMA"] = {"actual": test_series.values, "predicted": arima.predict(n_test)}
    except Exception as e:
        print(f"  ARIMA failed: {e}")

    # Random Forest
    print("  [4/5] Random Forest ...")
    rf = RandomForestForecaster(n_estimators=100)
    rf.fit(train_df, sales_col="sales")
    eval_results["RandomForest"] = {"actual": test_series.values, "predicted": rf.predict(n_test)}

    # XGBoost
    print("  [5/5] XGBoost ...")
    xgb = XGBoostForecaster()
    xgb.fit(train_df, sales_col="sales")
    eval_results["XGBoost"] = {"actual": test_series.values, "predicted": xgb.predict(n_test)}

    # Evaluate
    metrics_df = evaluate_all(eval_results)
    best_model_name = metrics_df.iloc[0]["Model"]
    print(f"\n  Best model: {best_model_name} (RMSE: {metrics_df.iloc[0]['RMSE']})")
    print(metrics_df.to_string(index=False))

    # Evaluation figures
    eval_figs = {
        "predictions_fig": plot_predictions(eval_results, test_df["date"], "sales"),
        "metrics_fig": plot_metrics_comparison(metrics_df),
        "residuals_fig": plot_residuals(eval_results, best_model_name)
    }

    # ── Step 6: Generate 30-Day Forecast ─────────────────────────
    print("\n▶ Step 6: Generating 30-day forecast ...")
    model_map = {
        "Naive": naive,
        "MovingAverage": ma,
        "ARIMA": arima if "ARIMA" in eval_results else ma,
        "RandomForest": rf,
        "XGBoost": xgb
    }
    best_model = model_map[best_model_name]

    # Refit best ML model on full data for forecasting
    if hasattr(best_model, "_feature_cols"):
        best_model.fit(df_features, sales_col="sales")
    else:
        best_model.fit(df_clean.set_index("date")["sales"])

    forecast_df = generate_forecast(best_model, df_features, horizon=30)

    # ── Step 7: Export Forecast CSV ───────────────────────────────
    print("\n▶ Step 7: Exporting forecast CSV ...")
    export_forecast(forecast_df, OUTPUT_FORECAST_DIR, best_model_name)

    # ── Step 8: Generate HTML Report ─────────────────────────────
    print("\n▶ Step 8: Generating HTML report ...")
    forecast_fig = plot_forecast(df_clean, forecast_df, best_model_name)
    report_path = os.path.join(OUTPUT_REPORT_DIR, "sales_forecast_report.html")
    generate_html_report(
        eda_results=eda_results,
        metrics_df=metrics_df,
        best_model_name=best_model_name,
        forecast_df=forecast_df,
        eval_figs=eval_figs,
        forecast_fig=forecast_fig,
        output_path=report_path
    )

    print("\n" + "="*60)
    print("  ✅ Pipeline Complete!")
    print(f"  📊 Report: {report_path}")
    print(f"  📁 Forecast CSV: {OUTPUT_FORECAST_DIR}/")
    print("="*60 + "\n")
    return report_path, forecast_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to sales CSV file")
    parser.add_argument("--store", type=str, default=None, help="Filter by store_id")
    parser.add_argument("--product", type=str, default=None, help="Filter by product_id")
    args = parser.parse_args()
    run(data_path=args.data, store_id=args.store, product_id=args.product)
