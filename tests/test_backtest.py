import numpy as np, pandas as pd
from backtest import run_backtest

def make_labeled_df(n=300):
    np.random.seed(1)
    close = 18000 + np.cumsum(np.random.randn(n) * 80)
    df = pd.DataFrame({
        "Close": close,
        "regime": np.random.choice([0, 1, 2], n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    return df

def test_backtest_returns_float():
    df = make_labeled_df()
    summary, pf = run_backtest(df)
    assert isinstance(summary.loc["Strategy", "Total Return (%)"], float)

def test_backtest_portfolio_positive():
    df = make_labeled_df()
    _, pf = run_backtest(df)
    assert pf.value().iloc[-1] > 0
