"""
evaluation.py — MAE, RMSE, MAPE metrics and comparison visualizations.
PRD 7.5
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948"]


def mae(actual, predicted): return float(np.mean(np.abs(actual - predicted)))
def rmse(actual, predicted): return float(np.sqrt(np.mean((actual - predicted) ** 2)))
def mape(actual, predicted):
    mask = actual != 0
    if mask.sum() == 0: return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def evaluate_all(results: dict) -> pd.DataFrame:
    rows = []
    for name, vals in results.items():
        a, p = np.array(vals["actual"]), np.array(vals["predicted"])
        rows.append({"Model": name, "MAE": round(mae(a, p), 4), "RMSE": round(rmse(a, p), 4), "MAPE (%)": round(mape(a, p), 2)})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def plot_predictions(results: dict, test_dates: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 5))
    actual = list(results.values())[0]["actual"]
    ax.plot(test_dates, actual, color="black", linewidth=2.2, label="Actual", zorder=5)
    for i, (name, vals) in enumerate(results.items()):
        ax.plot(test_dates, vals["predicted"], color=PALETTE[i % len(PALETTE)], linewidth=1.5, linestyle="--", alpha=0.85, label=name)
    ax.set_title("Model Predictions vs Actual — Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Sales"); ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE (%)"]):
        bars = ax.bar(metrics_df["Model"], metrics_df[metric], color=PALETTE[:len(metrics_df)], edgecolor="white")
        ax.set_title(metric, fontweight="bold"); ax.tick_params(axis="x", rotation=30)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); return fig


def plot_residuals(results: dict, best_model: str) -> plt.Figure:
    residuals = np.array(results[best_model]["actual"]) - np.array(results[best_model]["predicted"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(residuals, color=PALETTE[0], linewidth=0.9); axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title(f"Residuals — {best_model}", fontweight="bold"); axes[0].set_xlabel("Test Period (days)"); axes[0].set_ylabel("Actual − Predicted")
    axes[1].hist(residuals, bins=20, color=PALETTE[0], edgecolor="white")
    axes[1].set_title("Residual Distribution", fontweight="bold"); axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Frequency")
    plt.tight_layout(); return fig
