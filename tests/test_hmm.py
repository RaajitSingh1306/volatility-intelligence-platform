from hmm_model import fit_hmm
from features import build_features
from garch_model import add_garch_vol
import numpy as np, pandas as pd

def make_df(n=300):
    np.random.seed(0)
    close = 18000 + np.cumsum(np.random.randn(n) * 80)
    df = pd.DataFrame({
        "Open": close * 0.999, "High": close * 1.01,
        "Low": close * 0.99, "Close": close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna()

def test_hmm_regime_count():
    df = make_df()
    df = build_features(df)
    df, _ = add_garch_vol(df)
    df_out, hmm, _ = fit_hmm(df)
    assert df_out["regime"].nunique() == 3

def test_hmm_labels_valid():
    df = make_df()
    df = build_features(df)
    df, _ = add_garch_vol(df)
    df_out, _, _ = fit_hmm(df)
    assert set(df_out["regime"].unique()).issubset({0, 1, 2})

def test_hmm_no_nulls():
    df = make_df()
    df = build_features(df)
    df, _ = add_garch_vol(df)
    df_out, _, _ = fit_hmm(df)
    assert df_out["regime"].isnull().sum() == 0
