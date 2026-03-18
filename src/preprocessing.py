"""
Data Preprocessing & Feature Engineering
- Handle missing dates (fill / interpolate)
- Remove anomalies / outliers (IQR method)
- Ensure consistent daily frequency
- Engineer time-based, lag, and rolling features
"""
import pandas as pd
import numpy as np


def clean(df: pd.DataFrame, date_col: str = "date", sales_col: str = "sales") -> pd.DataFrame:
    """
    Clean the dataframe:
    1. Set date as index, ensure daily frequency (reindex + interpolate gaps)
    2. Remove outliers using IQR method
    3. Clip negative sales to 0
    """
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    # Reindex to full daily range and interpolate missing values
    full_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    df = df.set_index(date_col).reindex(full_range)
    df.index.name = date_col

    # Interpolate numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="time")

    # Fill non-numeric columns forward/backward
    non_numeric = df.select_dtypes(exclude=[np.number]).columns
    df[non_numeric] = df[non_numeric].ffill().bfill()

    df = df.reset_index()

    # Clip negative sales
    df[sales_col] = df[sales_col].clip(lower=0)

    # Remove outliers via IQR
    Q1 = df[sales_col].quantile(0.25)
    Q3 = df[sales_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3.0 * IQR
    upper = Q3 + 3.0 * IQR
    outliers = ((df[sales_col] < lower) | (df[sales_col] > upper)).sum()
    df[sales_col] = df[sales_col].clip(lower=lower, upper=upper)

    print(f"[Preprocessing] Clean complete | Rows: {len(df)} | Outliers capped: {outliers}")
    return df


def engineer_features(df: pd.DataFrame, date_col: str = "date", sales_col: str = "sales") -> pd.DataFrame:
    """
    Engineer time-based, lag, and rolling features.
    """
    df = df.copy().sort_values(date_col).reset_index(drop=True)
    dt = pd.to_datetime(df[date_col])

    # Time-based features
    df["day_of_week"] = dt.dt.dayofweek          # 0=Mon, 6=Sun
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)

    # Simple holiday indicator (weekends + common holiday months)
    df["holiday_indicator"] = ((df["is_weekend"] == 1) | (df["month"].isin([12, 1]))).astype(int)

    # Lag features
    df["lag_1"] = df[sales_col].shift(1)
    df["lag_7"] = df[sales_col].shift(7)
    df["lag_30"] = df[sales_col].shift(30)

    # Rolling features
    df["rolling_mean_7"] = df[sales_col].shift(1).rolling(window=7).mean()
    df["rolling_mean_30"] = df[sales_col].shift(1).rolling(window=30).mean()
    df["rolling_std_7"] = df[sales_col].shift(1).rolling(window=7).std()

    print(f"[Preprocessing] Feature engineering complete | Features: {df.shape[1]} columns")
    return df


def train_test_split_ts(df: pd.DataFrame, test_days: int = 30, date_col: str = "date"):
    """Split into train/test by time — last N days as test."""
    cutoff = df[date_col].max() - pd.Timedelta(days=test_days)
    train = df[df[date_col] <= cutoff].copy()
    test = df[df[date_col] > cutoff].copy()
    print(f"[Preprocessing] Train: {len(train)} rows | Test: {len(test)} rows | Cutoff: {cutoff.date()}")
    return train, test
