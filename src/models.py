"""
models.py — Forecasting models: Naive, MovingAverage, ARIMA, LinearRegression, RandomForest, XGBoost
PRD 7.4
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

RANDOM_SEED = 42


class NaiveForecaster:
    name = "Naive"
    def fit(self, train: pd.Series):
        self._last = float(train.iloc[-1]); return self
    def predict(self, n_periods: int) -> np.ndarray:
        return np.full(n_periods, self._last)


class MovingAverageForecaster:
    name = "MovingAverage"
    def __init__(self, window=7): self.window = window
    def fit(self, train: pd.Series):
        self._mean = float(train.iloc[-self.window:].mean()); return self
    def predict(self, n_periods: int) -> np.ndarray:
        return np.full(n_periods, self._mean)


class ARIMAForecaster:
    name = "ARIMA"
    def __init__(self, order=None): self.order = order; self._fit = None
    def fit(self, train: pd.Series):
        from statsmodels.tsa.arima.model import ARIMA
        if self.order:
            self._fit = ARIMA(train, order=self.order).fit(); return self
        best_aic, best_fit = np.inf, None
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        m = ARIMA(train, order=(p, d, q)).fit()
                        if m.aic < best_aic:
                            best_aic, best_fit = m.aic, m; self.order = (p, d, q)
                    except Exception: continue
        self._fit = best_fit
        print(f"[ARIMA] Best order: {self.order} | AIC: {round(best_aic, 2)}")
        return self
    def predict(self, n_periods: int) -> np.ndarray:
        return np.clip(self._fit.forecast(steps=n_periods).values, 0, None)


class MLForecaster:
    name = "ML"
    def __init__(self, model): self._model = model; self._feature_cols = None; self._last_window = None; self._sales_col = "sales"
    def _get_feature_cols(self, df, sales_col="sales"):
        exclude = {sales_col, "date", "store_id", "product_id"}
        return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    def fit(self, train_df, sales_col="sales"):
        self._sales_col = sales_col; self._feature_cols = self._get_feature_cols(train_df, sales_col)
        data = train_df.dropna(subset=self._feature_cols + [sales_col])
        self._model.fit(data[self._feature_cols], data[sales_col])
        self._last_window = train_df.tail(60).copy(); return self
    def predict(self, n_periods: int, **_) -> np.ndarray:
        from src.preprocessing import engineer_features
        history = self._last_window.copy(); preds = []
        for _ in range(n_periods):
            eng = engineer_features(history, sales_col=self._sales_col)
            last_row = eng.iloc[[-1]][self._feature_cols].fillna(0)
            pred = max(0.0, float(self._model.predict(last_row)[0]))
            preds.append(pred)
            next_date = pd.to_datetime(history["date"].iloc[-1]) + pd.Timedelta(days=1)
            new_row = history.iloc[[-1]].copy()
            new_row["date"] = next_date; new_row[self._sales_col] = pred
            history = pd.concat([history, new_row], ignore_index=True).tail(60)
        return np.array(preds)


class LinearRegressionForecaster(MLForecaster):
    name = "LinearRegression"
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        super().__init__(LinearRegression())


class RandomForestForecaster(MLForecaster):
    name = "RandomForest"
    def __init__(self, n_estimators=100):
        from sklearn.ensemble import RandomForestRegressor
        super().__init__(RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_SEED, n_jobs=-1))


class XGBoostForecaster(MLForecaster):
    name = "XGBoost"
    def __init__(self):
        from xgboost import XGBRegressor
        super().__init__(XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_SEED, verbosity=0))
