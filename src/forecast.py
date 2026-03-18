"""
forecast.py — Generate 30-day forward forecast, export to CSV, and plot.
PRD 7.6
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]


def generate_forecast(model, train_df, horizon=30, sales_col="sales", date_col="date") -> pd.DataFrame:
    last_date = pd.to_datetime(train_df[date_col].max())
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    preds = model.predict(horizon)
    return pd.DataFrame({"date": future_dates, "forecast": np.round(np.clip(preds, 0, None), 2)})


def export_forecast(forecast_df, output_dir, model_name) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fname = f"forecast_{model_name.lower().replace(' ', '_')}_30day.csv"
    filepath = os.path.join(output_dir, fname)
    forecast_df.to_csv(filepath, index=False)
    print(f"[Forecast] Exported → {filepath}")
    return filepath


def plot_forecast(train_df, forecast_df, model_name, sales_col="sales", date_col="date") -> plt.Figure:
    history = train_df.tail(90).copy()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(history[date_col], history[sales_col], color=PALETTE[0], linewidth=1.5, label="Historical Sales")
    ax.plot(forecast_df["date"], forecast_df["forecast"], color=PALETTE[1], linewidth=2.5, linestyle="--", label=f"{model_name} — 30-day Forecast")
    ax.fill_between(forecast_df["date"], forecast_df["forecast"] * 0.90, forecast_df["forecast"] * 1.10, alpha=0.20, color=PALETTE[1], label="±10% Band")
    ax.axvline(forecast_df["date"].iloc[0], color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"30-Day Sales Forecast — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Sales"); ax.legend()
    plt.tight_layout(); return fig
