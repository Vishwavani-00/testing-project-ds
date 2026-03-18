import numpy as np
import pandas as pd

np.random.seed(42)
dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
stores = [1, 2]
products = [1, 2, 3]
rows = []
base = {(s,p): 100 + s*20 + p*15 for s in stores for p in products}
for date in dates:
    dow = date.dayofweek
    month = date.month
    for s in stores:
        for p in products:
            trend = (date - dates[0]).days * 0.05
            seasonal = 30 * np.sin(2 * np.pi * date.dayofyear / 365)
            weekly = 20 if dow >= 5 else 0
            holiday = 50 if (month == 12 and date.day >= 20) or (month == 1 and date.day <= 5) else 0
            promo = np.random.choice([0, 1], p=[0.85, 0.15])
            promo_boost = promo * 40
            noise = np.random.normal(0, 15)
            sales = max(0, base[(s,p)] + trend + seasonal + weekly + holiday + promo_boost + noise)
            rows.append({"date": date.strftime("%Y-%m-%d"), "store_id": s, "product_id": p,
                         "sales": round(sales, 2), "promotions": int(promo)})

df = pd.DataFrame(rows)
df.to_csv("/tmp/retail-ds-fresh/data/retail_sales_raw.csv", index=False)
print(f"Generated {len(df)} rows")
print(df.head())
