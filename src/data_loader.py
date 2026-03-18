"""
Data Loader
Loads the Kaggle Store Sales / Retail Sales dataset and performs initial validation.
Expected columns: date, store_id, product_id, sales, promotions (optional)
"""
import os
import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and standardise column names."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Ensure required columns exist
    required = {"date", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Optional columns — add defaults if absent
    if "store_id" not in df.columns:
        df["store_id"] = "S1"
    if "product_id" not in df.columns:
        df["product_id"] = "P1"
    if "promotions" not in df.columns:
        df["promotions"] = 0

    print(f"[DataLoader] Loaded {len(df)} rows | Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def generate_sample_data(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic retail sales data for development/testing.
    Simulates trend + weekly seasonality + noise + occasional promotions.
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")

    trend = np.linspace(100, 160, n_days)
    weekly = 20 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    monthly = 10 * np.sin(2 * np.pi * np.arange(n_days) / 30)
    noise = np.random.normal(0, 8, n_days)
    promotions = np.random.binomial(1, 0.1, n_days)  # 10% days have promotions
    promo_boost = promotions * np.random.uniform(10, 30, n_days)

    sales = trend + weekly + monthly + noise + promo_boost
    sales = np.clip(sales, 0, None)

    df = pd.DataFrame({
        "date": dates,
        "store_id": "S1",
        "product_id": "P1",
        "sales": np.round(sales, 2),
        "promotions": promotions
    })
    print(f"[DataLoader] Generated {len(df)} rows of synthetic data.")
    return df
