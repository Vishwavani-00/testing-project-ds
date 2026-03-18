"""Unit tests for preprocessing module."""
import sys, os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import clean, engineer_features, train_test_split_ts
from src.data_loader   import generate_sample_data


@pytest.fixture
def raw_df():
    return generate_sample_data(n_days=120)


class TestClean:
    def test_no_negative_sales(self, raw_df):
        assert (clean(raw_df)["sales"] >= 0).all()

    def test_daily_frequency(self, raw_df):
        df   = clean(raw_df)
        diff = pd.to_datetime(df["date"]).diff().dropna()
        assert (diff == pd.Timedelta("1D")).all()

    def test_no_null_sales(self, raw_df):
        assert clean(raw_df)["sales"].isnull().sum() == 0


class TestFeatureEngineering:
    def test_lag_columns(self, raw_df):
        df = engineer_features(clean(raw_df))
        for col in ["lag_1", "lag_7", "lag_30"]:
            assert col in df.columns

    def test_rolling_columns(self, raw_df):
        df = engineer_features(clean(raw_df))
        for col in ["rolling_mean_7", "rolling_mean_30"]:
            assert col in df.columns

    def test_time_columns(self, raw_df):
        df = engineer_features(clean(raw_df))
        for col in ["day_of_week", "month", "is_weekend"]:
            assert col in df.columns


class TestTrainTestSplit:
    def test_test_size(self, raw_df):
        df = clean(raw_df)
        _, test = train_test_split_ts(df, test_days=20)
        assert len(test) == 20

    def test_no_overlap(self, raw_df):
        df       = clean(raw_df)
        train, test = train_test_split_ts(df, test_days=20)
        assert train["date"].max() < test["date"].min()
