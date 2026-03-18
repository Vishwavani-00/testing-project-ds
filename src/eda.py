"""
eda.py
======
Exploratory Data Analysis module (PRD 7.3):
  - Trend visualisation
  - Seasonality detection (day-of-week, monthly box plots)
  - Seasonal decomposition
  - Stationarity check (ADF test)
  - Feature correlation heatmap

All functions return matplotlib Figure objects for embedding in the HTML report.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948"]
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})


def plot_sales_trend(df: pd.DataFrame,
                     date_col: str = "date",
                     sales_col: str = "sales") -> plt.Figure:
    """Line chart of daily sales with 30-day rolling mean overlay."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[date_col], df[sales_col],
            color=PALETTE[0], alpha=0.55, linewidth=0.9, label="Daily Sales")
    ax.plot(df[date_col], df[sales_col].rolling(30).mean(),
            color=PALETTE[1], linewidth=2.2, label="30-Day Rolling Mean")
    ax.set_title("Sales Trend Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Sales")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig


def plot_seasonality(df: pd.DataFrame,
                     date_col: str = "date",
                     sales_col: str = "sales") -> plt.Figure:
    """Box plots of sales by day-of-week and by month."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["dow"]        = dt.dt.day_name()
    df["month_name"] = dt.dt.month_name()

    dow_order   = ["Monday","Tuesday","Wednesday","Thursday",
                   "Friday","Saturday","Sunday"]
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    sns.boxplot(data=df, x="dow", y=sales_col,
                order=dow_order, ax=axes[0], palette=PALETTE)
    axes[0].set_title("Sales by Day of Week", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)

    sns.boxplot(data=df[df["month_name"].isin(month_order)],
                x="month_name", y=sales_col,
                order=month_order, ax=axes[1], palette=PALETTE)
    axes[1].set_title("Sales by Month", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def plot_decomposition(df: pd.DataFrame,
                       date_col: str = "date",
                       sales_col: str = "sales",
                       period: int = 7) -> plt.Figure:
    """Additive seasonal decomposition into trend / seasonal / residual."""
    series = (df.set_index(date_col)[sales_col]
                .asfreq("D")
                .interpolate(method="time"))
    result = seasonal_decompose(series, model="additive",
                                period=period, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, component, label, color in zip(
        axes,
        [result.observed, result.trend, result.seasonal, result.resid],
        ["Observed", "Trend", "Seasonal", "Residual"],
        PALETTE
    ):
        component.plot(ax=ax, color=color, linewidth=0.9)
        ax.set_ylabel(label)

    fig.suptitle(f"Seasonal Decomposition (period={period} days)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller stationarity test.

    Returns:
        dict with adf_statistic, p_value, critical_values, is_stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    is_stationary = result[1] < 0.05
    output = {
        "adf_statistic":  round(result[0], 4),
        "p_value":        round(result[1], 4),
        "lags_used":      result[2],
        "n_observations": result[3],
        "critical_values": {k: round(v, 4) for k, v in result[4].items()},
        "is_stationary":  is_stationary,
    }
    verdict = "STATIONARY ✓" if is_stationary else "NON-STATIONARY ✗"
    print(f"[EDA] ADF p-value: {output['p_value']} → {verdict}")
    return output


def plot_correlation(df: pd.DataFrame,
                     sales_col: str = "sales") -> plt.Figure:
    """Horizontal bar chart of feature correlations with sales."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = (df[num_cols].corr()[[sales_col]]
                        .drop(sales_col)
                        .sort_values(sales_col))

    fig, ax = plt.subplots(figsize=(5, max(4, len(corr) * 0.35)))
    colors = [PALETTE[0] if v >= 0 else PALETTE[3] for v in corr[sales_col]]
    ax.barh(corr.index, corr[sales_col], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Feature Correlation with {sales_col}", fontweight="bold")
    ax.set_xlabel("Pearson r")
    plt.tight_layout()
    return fig


def run_eda(df: pd.DataFrame,
            date_col: str = "date",
            sales_col: str = "sales") -> dict:
    """
    Run full EDA suite and return results dict.

    Returns:
        {trend_fig, seasonality_fig, decomposition_fig, correlation_fig,
         adf_stats, summary_stats}
    """
    print("[EDA] Running EDA ...")
    return {
        "trend_fig":        plot_sales_trend(df, date_col, sales_col),
        "seasonality_fig":  plot_seasonality(df, date_col, sales_col),
        "decomposition_fig": plot_decomposition(df, date_col, sales_col),
        "correlation_fig":  plot_correlation(df, sales_col),
        "adf_stats":        adf_test(df[sales_col]),
        "summary_stats":    df[sales_col].describe().round(2).to_dict(),
    }
