"""
forecast.py
===========
Generate 30-day forward forecast using the best model (PRD 7.6):
  - Forecast horizon: 30 days
  - Export predictions to CSV
  - Return forecast visualisation figure
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FORECAST_HORIZON = 30
PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]


def generate_forecast(model,
                      train_df: pd.DataFrame,
                      horizon: int = FORECAST_HORIZON,
                      sales_col: str = "sales",
                      date_col: str = "date") -> pd.DataFrame:
    """
    Generate a horizon-day forecast from the fitted model.

    Args:
        model:     Fitted forecaster instance.
        train_df:  Full training DataFrame (for last-window context).
        horizon:   Number of days to forecast.
        sales_col: Sales column name.
        date_col:  Date column name.

    Returns:
        DataFrame with columns: date, forecast.
    """
    last_date    = pd.to_datetime(train_df[date_col].max())
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                 periods=horizon, freq="D")

    # ML models expose _feature_cols; others use simple predict(n)
    if hasattr(model, "_feature_cols") and model._feature_cols is not None:
        preds = model.predict(horizon)
    else:
        preds = model.predict(horizon)

    return pd.DataFrame({
        "date":     future_dates,
        "forecast": np.round(np.clip(preds, 0, None), 2),
    })


def export_forecast(forecast_df: pd.DataFrame,
                    output_dir: str,
                    model_name: str) -> str:
    """
    Export forecast to CSV.

    Returns:
        Absolute path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    fname    = f"forecast_{model_name.lower().replace(' ', '_')}_30day.csv"
    filepath = os.path.join(output_dir, fname)
    forecast_df.to_csv(filepath, index=False)
    print(f"[Forecast] Exported → {filepath}")
    return filepath


def plot_forecast(train_df: pd.DataFrame,
                  forecast_df: pd.DataFrame,
                  model_name: str,
                  sales_col: str = "sales",
                  date_col: str = "date") -> plt.Figure:
    """
    Plot last 90 days of history + 30-day forecast with ±10% band.
    """
    history = train_df.tail(90).copy()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(history[date_col], history[sales_col],
            color=PALETTE[0], linewidth=1.5, label="Historical Sales")
    ax.plot(forecast_df["date"], forecast_df["forecast"],
            color=PALETTE[1], linewidth=2.5, linestyle="--",
            label=f"{model_name} — 30-day Forecast")

    # ±10 % confidence band
    ax.fill_between(forecast_df["date"],
                    forecast_df["forecast"] * 0.90,
                    forecast_df["forecast"] * 1.10,
                    alpha=0.20, color=PALETTE[1], label="±10% Band")

    ax.axvline(forecast_df["date"].iloc[0], color="gray",
               linestyle=":", linewidth=1)
    ax.set_title(f"30-Day Sales Forecast — {model_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Sales")
    ax.legend()
    plt.tight_layout()
    return fig
