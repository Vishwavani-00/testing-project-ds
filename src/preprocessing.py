"""
preprocessing.py
================
Data cleaning, feature engineering, and train/test split utilities.

Cleaning rules (from PRD 7.1):
  - Fill missing dates / interpolate gaps
  - Remove anomalies / outliers (IQR method)
  - Ensure consistent daily frequency
  - Clip negative sales to 0

Feature engineering (from PRD 7.2):
  - Time-based: day_of_week, month, quarter, year, is_weekend, holiday_indicator
  - Lag features: lag_1, lag_7, lag_30
  - Rolling metrics: rolling_mean_7, rolling_mean_30, rolling_std_7
"""
import pandas as pd
import numpy as np

RANDOM_SEED = 42


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame,
          date_col: str = "date",
          sales_col: str = "sales") -> pd.DataFrame:
    """
    Clean the sales DataFrame:
      1. Sort by date
      2. Reindex to full daily range and interpolate missing values
      3. Clip negative sales to 0
      4. Cap outliers using 3×IQR method

    Args:
        df:        Input DataFrame.
        date_col:  Name of the date column.
        sales_col: Name of the sales column.

    Returns:
        Cleaned DataFrame with continuous daily index.
    """
    df = df.copy().sort_values(date_col).reset_index(drop=True)

    # Reindex to full daily range
    full_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    df = df.set_index(date_col).reindex(full_range)
    df.index.name = date_col

    # Interpolate numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method="time")

    # Forward/back-fill categorical columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    df[cat_cols] = df[cat_cols].ffill().bfill()

    df = df.reset_index()

    # Clip negative sales
    df[sales_col] = df[sales_col].clip(lower=0)

    # Cap outliers via IQR (3× for soft capping)
    q1, q3 = df[sales_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    n_outliers = ((df[sales_col] < lower) | (df[sales_col] > upper)).sum()
    df[sales_col] = df[sales_col].clip(lower=lower, upper=upper)

    print(f"[Preprocessing] Cleaned: {len(df)} rows | Outliers capped: {n_outliers}")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame,
                      date_col: str = "date",
                      sales_col: str = "sales") -> pd.DataFrame:
    """
    Engineer time-based, lag, and rolling features.

    Args:
        df:        Cleaned DataFrame.
        date_col:  Name of the date column.
        sales_col: Name of the sales column.

    Returns:
        DataFrame with additional feature columns.
    """
    df = df.copy().sort_values(date_col).reset_index(drop=True)
    dt = pd.to_datetime(df[date_col])

    # ── Time-based features ──
    df["day_of_week"]   = dt.dt.dayofweek          # 0=Mon … 6=Sun
    df["day_of_month"]  = dt.dt.day
    df["month"]         = dt.dt.month
    df["quarter"]       = dt.dt.quarter
    df["year"]          = dt.dt.year
    df["week_of_year"]  = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"]    = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"]   = dt.dt.is_month_end.astype(int)

    # Simple holiday indicator (weekends + Dec/Jan)
    df["holiday_indicator"] = (
        (df["is_weekend"] == 1) | (df["month"].isin([12, 1]))
    ).astype(int)

    # ── Lag features ──
    df["lag_1"]  = df[sales_col].shift(1)
    df["lag_7"]  = df[sales_col].shift(7)
    df["lag_30"] = df[sales_col].shift(30)

    # ── Rolling metrics ──
    df["rolling_mean_7"]  = df[sales_col].shift(1).rolling(7).mean()
    df["rolling_mean_30"] = df[sales_col].shift(1).rolling(30).mean()
    df["rolling_std_7"]   = df[sales_col].shift(1).rolling(7).std()

    print(f"[Preprocessing] Feature engineering complete | "
          f"{df.shape[1]} columns")
    return df


# ── Train / Test Split ────────────────────────────────────────────────────────

def train_test_split_ts(df: pd.DataFrame,
                        test_days: int = 30,
                        date_col: str = "date"):
    """
    Time-aware split — last N days as test, rest as train.

    Args:
        df:        Feature-engineered DataFrame.
        test_days: Number of days to hold out as test.
        date_col:  Name of the date column.

    Returns:
        (train_df, test_df)
    """
    cutoff = df[date_col].max() - pd.Timedelta(days=test_days)
    train  = df[df[date_col] <= cutoff].copy()
    test   = df[df[date_col] >  cutoff].copy()
    print(f"[Preprocessing] Train: {len(train)} rows | "
          f"Test: {len(test)} rows | Cutoff: {cutoff.date()}")
    return train, test
