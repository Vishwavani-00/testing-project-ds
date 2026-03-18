"""
Exploratory Data Analysis (EDA)
- Trend visualization
- Seasonality detection
- Stationarity checks (ADF test)
Returns matplotlib figures for embedding in the HTML report.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948"]
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})


def plot_sales_trend(df: pd.DataFrame, date_col: str = "date", sales_col: str = "sales") -> plt.Figure:
    """Line plot of daily sales with 30-day rolling mean."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[date_col], df[sales_col], color=PALETTE[0], alpha=0.6, linewidth=0.8, label="Daily Sales")
    rolling = df[sales_col].rolling(30).mean()
    ax.plot(df[date_col], rolling, color=PALETTE[1], linewidth=2, label="30-Day Rolling Mean")
    ax.set_title("Sales Trend Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_seasonality(df: pd.DataFrame, date_col: str = "date", sales_col: str = "sales") -> plt.Figure:
    """Box plots by day-of-week and by month."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["dow"] = dt.dt.day_name()
    df["month_name"] = dt.dt.month_name()
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.boxplot(data=df, x="dow", y=sales_col, order=dow_order, ax=axes[0], palette=PALETTE)
    axes[0].set_title("Sales by Day of Week", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)

    month_data = df[df["month_name"].isin(month_order)]
    sns.boxplot(data=month_data, x="month_name", y=sales_col, order=month_order, ax=axes[1], palette=PALETTE)
    axes[1].set_title("Sales by Month", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def plot_decomposition(df: pd.DataFrame, date_col: str = "date", sales_col: str = "sales", period: int = 7) -> plt.Figure:
    """Seasonal decomposition plot (additive)."""
    series = df.set_index(date_col)[sales_col].asfreq("D").fillna(method="ffill")
    result = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    result.observed.plot(ax=axes[0], color=PALETTE[0])
    axes[0].set_ylabel("Observed")
    result.trend.plot(ax=axes[1], color=PALETTE[1])
    axes[1].set_ylabel("Trend")
    result.seasonal.plot(ax=axes[2], color=PALETTE[2])
    axes[2].set_ylabel("Seasonal")
    result.resid.plot(ax=axes[3], color=PALETTE[3])
    axes[3].set_ylabel("Residual")

    fig.suptitle("Seasonal Decomposition (Period=7 days)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller stationarity test.
    Returns dict with ADF stat, p-value, critical values, and verdict.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    output = {
        "adf_statistic": round(result[0], 4),
        "p_value": round(result[1], 4),
        "lags_used": result[2],
        "n_observations": result[3],
        "critical_values": {k: round(v, 4) for k, v in result[4].items()},
        "is_stationary": result[1] < 0.05
    }
    verdict = "STATIONARY ✓" if output["is_stationary"] else "NON-STATIONARY ✗ (differencing may be needed)"
    print(f"[EDA] ADF Test | p-value: {output['p_value']} | {verdict}")
    return output


def plot_correlation(df: pd.DataFrame, sales_col: str = "sales") -> plt.Figure:
    """Correlation heatmap of numeric features vs sales."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if sales_col in num_cols:
        # Reorder so sales is first
        num_cols = [sales_col] + [c for c in num_cols if c != sales_col]
    corr = df[num_cols].corr()[[sales_col]].drop(sales_col)

    fig, ax = plt.subplots(figsize=(4, max(4, len(corr) * 0.4)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title(f"Feature Correlation with {sales_col}", fontweight="bold")
    plt.tight_layout()
    return fig


def run_eda(df: pd.DataFrame, date_col: str = "date", sales_col: str = "sales") -> dict:
    """Run full EDA and return dict of figures + stats."""
    print("[EDA] Running full EDA ...")
    results = {}
    results["trend_fig"] = plot_sales_trend(df, date_col, sales_col)
    results["seasonality_fig"] = plot_seasonality(df, date_col, sales_col)
    results["decomposition_fig"] = plot_decomposition(df, date_col, sales_col)
    results["adf_stats"] = adf_test(df[sales_col])
    results["summary_stats"] = df[sales_col].describe().round(2).to_dict()
    print("[EDA] EDA complete.")
    return results
