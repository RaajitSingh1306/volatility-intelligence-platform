import pytest
import pandas as pd
import numpy as np
from features import build_features
from garch_model import add_garch_vol

@pytest.fixture
def sample_df():
    n = 300
    np.random.seed(42)
    close = 18000 + np.cumsum(np.random.randn(n) * 80)
    df = pd.DataFrame({
        "Open": close * 0.999, "High": close * 1.01,
        "Low": close * 0.99, "Close": close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna()

def test_garch_output_shape(sample_df):
    df = build_features(sample_df)
    df_out, garch = add_garch_vol(df)
    assert "garch_vol" in df_out.columns

def test_garch_no_nans(sample_df):
    df = build_features(sample_df)
    df_out, _ = add_garch_vol(df)
    assert df_out["garch_vol"].dropna().shape[0] > 0

def test_garch_vol_positive(sample_df):
    df = build_features(sample_df)
    df_out, _ = add_garch_vol(df)
    assert (df_out["garch_vol"].dropna() > 0).all()
