import pandas as pd
import numpy as np

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["store_id","product_id","date"]).reset_index(drop=True)
    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    parts = []
    for (s,p), grp in df.groupby(["store_id","product_id"]):
        grp = grp.set_index("date").reindex(all_dates).rename_axis("date").reset_index()
        grp["store_id"] = s
        grp["product_id"] = p
        grp["sales"] = grp["sales"].interpolate(method="linear")
        grp["promotions"] = grp["promotions"].fillna(0)
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = (((df["month"]==12)&(df["date"].dt.day>=20)) |
                        ((df["month"]==1)&(df["date"].dt.day<=5))).astype(int)
    for lag in [1, 7, 30]:
        df[f"lag_{lag}"] = df.groupby(["store_id","product_id"])["sales"].shift(lag)
    for w in [7, 30]:
        df[f"ma_{w}"] = df.groupby(["store_id","product_id"])["sales"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    return df
