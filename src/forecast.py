"""
Forecast Generation
- Generate 30-day forward forecast using the best model
- Export predictions to CSV
- Return forecast plot figure
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
FORECAST_HORIZON = 30


def generate_forecast(model, train_df: pd.DataFrame, horizon: int = FORECAST_HORIZON,
                      sales_col: str = "sales", date_col: str = "date") -> pd.DataFrame:
    """
    Generate horizon-day forecast using the fitted model.
    Returns a DataFrame with columns: date, forecast.
    """
    last_date = pd.to_datetime(train_df[date_col].max())
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # ML models use recursive predict; baseline/statistical use simple predict
    if hasattr(model, "_feature_cols") and model._feature_cols is not None:
        preds = model.predict(horizon, future_dates=future_dates)
    else:
        preds = model.predict(horizon)

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": np.round(np.clip(preds, 0, None), 2)
    })
    return forecast_df


def export_forecast(forecast_df: pd.DataFrame, output_dir: str, model_name: str) -> str:
    """Export forecast DataFrame to CSV. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"forecast_{model_name.lower().replace(' ', '_')}_30day.csv"
    filepath = os.path.join(output_dir, filename)
    forecast_df.to_csv(filepath, index=False)
    print(f"[Forecast] Exported → {filepath}")
    return filepath


def plot_forecast(train_df: pd.DataFrame, forecast_df: pd.DataFrame,
                  model_name: str, sales_col: str = "sales",
                  date_col: str = "date") -> plt.Figure:
    """Plot historical sales + 30-day forecast with confidence band."""
    # Show last 90 days of history
    history = train_df.tail(90).copy()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(history[date_col], history[sales_col], color=PALETTE[0],
            linewidth=1.5, label="Historical Sales")
    ax.plot(forecast_df["date"], forecast_df["forecast"], color=PALETTE[1],
            linewidth=2.5, linestyle="--", label=f"{model_name} Forecast (30 days)")

    # Simple confidence band (±10% of forecast)
    upper = forecast_df["forecast"] * 1.10
    lower = forecast_df["forecast"] * 0.90
    ax.fill_between(forecast_df["date"], lower, upper, alpha=0.2, color=PALETTE[1], label="±10% Band")

    # Vertical line at forecast start
    ax.axvline(x=forecast_df["date"].iloc[0], color="gray", linestyle=":", linewidth=1)

    ax.set_title(f"30-Day Sales Forecast — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.tight_layout()
    return fig
