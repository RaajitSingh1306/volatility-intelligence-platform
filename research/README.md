# Research

One notebook that motivates modelling volatility instead of returns.

## 01_arima_null_result.ipynb

- Augmented Dickey-Fuller test on Nifty 50 price levels and on log returns.
- ARIMA(1,0,1) on daily log returns, to check whether the AR and MA terms are significant.
- Engle ARCH-LM test and ACF of squared returns, to check for volatility clustering.
- Reads `../data/nifty_ohlcv.csv`. Statistics are in the saved cell outputs of the notebook, not copied here.
- It tests this one series only. It does not show that returns are unpredictable in general.

A second notebook on sector momentum was removed: its table was generated with `np.random.normal`, not market data.
Sector momentum lives in the separate Nifty Sector Rotation repository.
