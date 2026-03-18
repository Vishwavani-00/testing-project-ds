"""
preprocessing.py — Data cleaning and feature engineering.
PRD 7.1 Cleaning: fill missing dates, cap outliers, ensure daily frequency.
PRD 7.2 Features: time-based, lag (1,7,30), rolling mean (7,30).
"""
import pandas as pd
import numpy as np

RANDOM_SEED = 42


def clean(df: pd.DataFrame, date_col="date", sales_col="sales") -> pd.DataFrame:
    df = df.copy().sort_values(date_col).reset_index(drop=True)
    full_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    df = df.set_index(date_col).reindex(full_range)
    df.index.name = date_col
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method="time")
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    df[cat_cols] = df[cat_cols].ffill().bfill()
    df = df.reset_index()
    df[sales_col] = df[sales_col].clip(lower=0)
    q1, q3 = df[sales_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    n_out = ((df[sales_col] < lower) | (df[sales_col] > upper)).sum()
    df[sales_col] = df[sales_col].clip(lower=lower, upper=upper)
    print(f"[Preprocessing] Cleaned: {len(df)} rows | Outliers capped: {n_out}")
    return df


def engineer_features(df: pd.DataFrame, date_col="date", sales_col="sales") -> pd.DataFrame:
    df = df.copy().sort_values(date_col).reset_index(drop=True)
    dt = pd.to_datetime(df[date_col])
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["holiday_indicator"] = ((df["is_weekend"] == 1) | (df["month"].isin([12, 1]))).astype(int)
    df["lag_1"] = df[sales_col].shift(1)
    df["lag_7"] = df[sales_col].shift(7)
    df["lag_30"] = df[sales_col].shift(30)
    df["rolling_mean_7"] = df[sales_col].shift(1).rolling(7).mean()
    df["rolling_mean_30"] = df[sales_col].shift(1).rolling(30).mean()
    df["rolling_std_7"] = df[sales_col].shift(1).rolling(7).std()
    print(f"[Preprocessing] Features engineered | {df.shape[1]} columns")
    return df


def train_test_split_ts(df: pd.DataFrame, test_days=30, date_col="date"):
    cutoff = df[date_col].max() - pd.Timedelta(days=test_days)
    train = df[df[date_col] <= cutoff].copy()
    test = df[df[date_col] > cutoff].copy()
    print(f"[Preprocessing] Train: {len(train)} | Test: {len(test)} | Cutoff: {cutoff.date()}")
    return train, test
