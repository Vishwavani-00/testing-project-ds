"""Unit tests for forecasting models."""
import sys, os
import pytest
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import NaiveForecaster, MovingAverageForecaster
from src.data_loader import generate_sample_data
from src.preprocessing import clean, engineer_features


@pytest.fixture
def train_series():
    df = generate_sample_data(n_days=100)
    df = clean(df)
    return df.set_index("date")["sales"]


@pytest.fixture
def train_df():
    df = generate_sample_data(n_days=100)
    df = clean(df)
    return engineer_features(df)


class TestNaive:
    def test_output_length(self, train_series):
        model = NaiveForecaster().fit(train_series)
        preds = model.predict(10)
        assert len(preds) == 10

    def test_constant_value(self, train_series):
        model = NaiveForecaster().fit(train_series)
        preds = model.predict(5)
        assert np.allclose(preds, preds[0])


class TestMovingAverage:
    def test_output_length(self, train_series):
        model = MovingAverageForecaster(window=7).fit(train_series)
        preds = model.predict(10)
        assert len(preds) == 10

    def test_no_negatives(self, train_series):
        model = MovingAverageForecaster().fit(train_series)
        preds = model.predict(30)
        assert (preds >= 0).all()
