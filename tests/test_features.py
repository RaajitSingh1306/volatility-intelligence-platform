"""
tests/test_features.py
-----------------------
Unit tests for feature engineering module.
"""

import numpy as np
import pandas as pd
import pytest

from features import build_features, garman_klass_vol, parkinson_vol, FEATURE_COLS


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Minimal synthetic OHLCV DataFrame for testing."""
    n = 150
    np.random.seed(0)
    close = 10_000 + np.cumsum(np.random.randn(n) * 50)
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, n),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )
    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df


def test_garman_klass_vol_shape(sample_ohlcv):
    gk = garman_klass_vol(sample_ohlcv, window=21)
    assert isinstance(gk, pd.Series)
    assert len(gk) == len(sample_ohlcv)


def test_parkinson_vol_positive(sample_ohlcv):
    pk = parkinson_vol(sample_ohlcv, window=21)
    assert (pk.dropna() >= 0).all()


def test_build_features_columns(sample_ohlcv):
    feat = build_features(sample_ohlcv)
    for col in FEATURE_COLS:
        assert col in feat.columns, f"Missing feature column: {col}"


def test_build_features_no_nulls(sample_ohlcv):
    feat = build_features(sample_ohlcv)
    assert feat[FEATURE_COLS].isnull().sum().sum() == 0


def test_build_features_row_count(sample_ohlcv):
    feat = build_features(sample_ohlcv)
    # Should have fewer rows due to dropna after rolling windows
    assert len(feat) < len(sample_ohlcv)
    assert len(feat) > 0