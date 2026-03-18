"""Unit tests for preprocessing module."""
import sys, os
import pytest
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import clean, engineer_features, train_test_split_ts
from src.data_loader import generate_sample_data


@pytest.fixture
def sample_df():
    return generate_sample_data(n_days=100)


class TestClean:
    def test_no_negative_sales(self, sample_df):
        result = clean(sample_df)
        assert (result["sales"] >= 0).all()

    def test_daily_frequency(self, sample_df):
        result = clean(sample_df)
        dates = pd.to_datetime(result["date"])
        diffs = dates.diff().dropna()
        assert (diffs == pd.Timedelta("1D")).all()

    def test_no_nulls_in_sales(self, sample_df):
        result = clean(sample_df)
        assert result["sales"].isnull().sum() == 0


class TestFeatureEngineering:
    def test_lag_features_created(self, sample_df):
        df = clean(sample_df)
        result = engineer_features(df)
        for col in ["lag_1", "lag_7", "lag_30"]:
            assert col in result.columns

    def test_rolling_features_created(self, sample_df):
        df = clean(sample_df)
        result = engineer_features(df)
        for col in ["rolling_mean_7", "rolling_mean_30"]:
            assert col in result.columns

    def test_time_features_created(self, sample_df):
        df = clean(sample_df)
        result = engineer_features(df)
        for col in ["day_of_week", "month", "year", "is_weekend"]:
            assert col in result.columns


class TestTrainTestSplit:
    def test_split_sizes(self, sample_df):
        df = clean(sample_df)
        train, test = train_test_split_ts(df, test_days=20)
        assert len(test) == 20
        assert len(train) == len(df) - 20

    def test_no_date_overlap(self, sample_df):
        df = clean(sample_df)
        train, test = train_test_split_ts(df, test_days=20)
        assert train["date"].max() < test["date"].min()
