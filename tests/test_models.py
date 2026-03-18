"""Unit tests for forecasting models."""
import sys, os, pytest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import NaiveForecaster, MovingAverageForecaster
from src.data_loader import generate_sample_data
from src.preprocessing import clean

@pytest.fixture
def series():
    df = generate_sample_data(n_days=100)
    return clean(df).set_index("date")["sales"]

class TestNaive:
    def test_length(self, series): assert len(NaiveForecaster().fit(series).predict(10)) == 10
    def test_constant(self, series): preds = NaiveForecaster().fit(series).predict(5); assert np.allclose(preds, preds[0])
    def test_no_negatives(self, series): assert (NaiveForecaster().fit(series).predict(30) >= 0).all()

class TestMovingAverage:
    def test_length(self, series): assert len(MovingAverageForecaster().fit(series).predict(10)) == 10
    def test_no_negatives(self, series): assert (MovingAverageForecaster().fit(series).predict(30) >= 0).all()
