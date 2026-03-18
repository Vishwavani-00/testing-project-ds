"""
Forecasting Models
Baseline: Naive, Moving Average
Statistical: ARIMA/SARIMA
ML: Linear Regression, Random Forest, XGBoost
All models implement a common interface: fit(train) → predict(n_periods) → pd.Series
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42


# ── Baseline Models ────────────────────────────────────────────────────────────

class NaiveForecaster:
    """Naive: last known value repeated for all future steps."""
    name = "Naive"

    def fit(self, train: pd.Series):
        self._last = train.iloc[-1]
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        return np.full(n_periods, self._last)


class MovingAverageForecaster:
    """Moving Average: mean of last N observations."""
    name = "MovingAverage"

    def __init__(self, window: int = 7):
        self.window = window

    def fit(self, train: pd.Series):
        self._mean = train.iloc[-self.window:].mean()
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        return np.full(n_periods, self._mean)


# ── Statistical Models ────────────────────────────────────────────────────────

class ARIMAForecaster:
    """ARIMA with auto order selection (tries a few combinations, picks best AIC)."""
    name = "ARIMA"

    def __init__(self, order: tuple = None):
        self.order = order
        self._model_fit = None

    def fit(self, train: pd.Series):
        from statsmodels.tsa.arima.model import ARIMA
        if self.order is not None:
            model = ARIMA(train, order=self.order)
            self._model_fit = model.fit()
        else:
            best_aic = np.inf
            best_fit = None
            for p in [0, 1, 2]:
                for d in [0, 1]:
                    for q in [0, 1, 2]:
                        try:
                            m = ARIMA(train, order=(p, d, q)).fit()
                            if m.aic < best_aic:
                                best_aic = m.aic
                                best_fit = m
                                self.order = (p, d, q)
                        except Exception:
                            continue
            self._model_fit = best_fit
            print(f"[ARIMA] Best order: {self.order} | AIC: {round(best_aic, 2)}")
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        forecast = self._model_fit.forecast(steps=n_periods)
        return np.clip(forecast.values, 0, None)


# ── ML Models ─────────────────────────────────────────────────────────────────

class MLForecaster:
    """
    Base class for ML-based forecasters.
    Uses lag + rolling features for supervised learning.
    """
    name = "ML"

    def __init__(self, model):
        self._model = model
        self._feature_cols = None
        self._last_window = None

    def _get_feature_cols(self, df: pd.DataFrame, sales_col: str = "sales") -> list:
        exclude = {sales_col, "date", "store_id", "product_id"}
        return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    def fit(self, train_df: pd.DataFrame, sales_col: str = "sales"):
        self._feature_cols = self._get_feature_cols(train_df, sales_col)
        data = train_df.dropna(subset=self._feature_cols + [sales_col])
        X = data[self._feature_cols]
        y = data[sales_col]
        self._model.fit(X, y)
        # Store last 30 rows for recursive forecasting
        self._last_window = train_df.tail(30).copy()
        self._sales_col = sales_col
        return self

    def predict(self, n_periods: int, future_dates: pd.DatetimeIndex = None) -> np.ndarray:
        """Recursive one-step-ahead forecasting."""
        from src.preprocessing import engineer_features
        history = self._last_window.copy()
        preds = []

        for i in range(n_periods):
            hist_eng = engineer_features(history, sales_col=self._sales_col)
            last_row = hist_eng.iloc[[-1]][self._feature_cols].fillna(0)
            pred = float(self._model.predict(last_row)[0])
            pred = max(0, pred)
            preds.append(pred)

            # Append predicted row to history
            next_date = history["date"].iloc[-1] + pd.Timedelta(days=1)
            new_row = history.iloc[[-1]].copy()
            new_row["date"] = next_date
            new_row[self._sales_col] = pred
            history = pd.concat([history, new_row], ignore_index=True).tail(60)

        return np.array(preds)


class LinearRegressionForecaster(MLForecaster):
    name = "LinearRegression"

    def __init__(self):
        from sklearn.linear_model import LinearRegression
        super().__init__(LinearRegression())


class RandomForestForecaster(MLForecaster):
    name = "RandomForest"

    def __init__(self, n_estimators: int = 100):
        from sklearn.ensemble import RandomForestRegressor
        super().__init__(RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_SEED))


class XGBoostForecaster(MLForecaster):
    name = "XGBoost"

    def __init__(self):
        from xgboost import XGBRegressor
        super().__init__(XGBRegressor(n_estimators=200, learning_rate=0.05,
                                      max_depth=4, random_state=RANDOM_SEED,
                                      verbosity=0))
