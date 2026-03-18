"""
Model Evaluation
Metrics: MAE, RMSE, MAPE
Comparison table + visualizations
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948"]


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def evaluate_all(results: dict) -> pd.DataFrame:
    """
    Given results dict {model_name: {"actual": arr, "predicted": arr}},
    return a DataFrame with MAE, RMSE, MAPE for each model.
    """
    rows = []
    for name, vals in results.items():
        actual = np.array(vals["actual"])
        predicted = np.array(vals["predicted"])
        rows.append({
            "Model": name,
            "MAE": round(mae(actual, predicted), 4),
            "RMSE": round(rmse(actual, predicted), 4),
            "MAPE (%)": round(mape(actual, predicted), 2)
        })
    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return df


def plot_predictions(results: dict, test_dates: pd.Series, sales_col: str = "sales") -> plt.Figure:
    """Line plot comparing all model predictions vs actuals on the test set."""
    fig, ax = plt.subplots(figsize=(13, 5))
    actual = list(results.values())[0]["actual"]
    ax.plot(test_dates, actual, color="black", linewidth=2, label="Actual", zorder=5)

    for i, (name, vals) in enumerate(results.items()):
        ax.plot(test_dates, vals["predicted"], color=PALETTE[i % len(PALETTE)],
                linewidth=1.5, linestyle="--", alpha=0.85, label=name)

    ax.set_title("Model Predictions vs Actual (Test Set)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing MAE, RMSE, MAPE across models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["MAE", "RMSE", "MAPE (%)"]
    labels = ["Mean Absolute Error", "Root Mean Squared Error", "MAPE (%)"]

    for ax, metric, label in zip(axes, metrics, labels):
        bars = ax.bar(metrics_df["Model"], metrics_df[metric],
                      color=PALETTE[:len(metrics_df)], edgecolor="white")
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return fig


def plot_residuals(results: dict, best_model: str) -> plt.Figure:
    """Residual plot for the best model."""
    vals = results[best_model]
    actual = np.array(vals["actual"])
    predicted = np.array(vals["predicted"])
    residuals = actual - predicted

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(residuals, color=PALETTE[0], linewidth=0.8)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title(f"Residuals — {best_model}", fontweight="bold")
    axes[0].set_xlabel("Test Period (days)")
    axes[0].set_ylabel("Residual")

    axes[1].hist(residuals, bins=20, color=PALETTE[0], edgecolor="white")
    axes[1].set_title("Residual Distribution", fontweight="bold")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    return fig
