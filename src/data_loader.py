"""
data_loader.py — Load CSV or generate synthetic retail sales data.
Expected CSV columns: date, sales (optional: store_id, product_id, promotions)
"""
import os
import pandas as pd
import numpy as np

RANDOM_SEED = 42


def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    required = {"date", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "store_id" not in df.columns:
        df["store_id"] = "S1"
    if "product_id" not in df.columns:
        df["product_id"] = "P1"
    if "promotions" not in df.columns:
        df["promotions"] = 0
    print(f"[DataLoader] Loaded {len(df)} rows | {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def generate_sample_data(n_days: int = 730, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate 2-year synthetic retail sales with trend + seasonality + noise."""
    np.random.seed(seed)
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")
    trend = np.linspace(100, 160, n_days)
    weekly = 20 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    monthly = 10 * np.sin(2 * np.pi * np.arange(n_days) / 30)
    noise = np.random.normal(0, 8, n_days)
    promo = np.random.binomial(1, 0.10, n_days)
    promo_boost = promo * np.random.uniform(10, 30, n_days)
    sales = np.clip(trend + weekly + monthly + noise + promo_boost, 0, None)
    df = pd.DataFrame({
        "date": dates, "store_id": "S1", "product_id": "P1",
        "sales": np.round(sales, 2), "promotions": promo,
    })
    print(f"[DataLoader] Generated {len(df)} rows of synthetic data.")
    return df
