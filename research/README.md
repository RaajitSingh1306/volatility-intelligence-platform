# Research

Academic and empirical underpinning for the Volatility Intelligence Platform.

## Contents

1. **`01_arima_null_result.ipynb` — The Return Unpredictability Baseline**
   - Implements Augmented Dickey-Fuller (ADF) tests establishing non-stationarity of Nifty 50 price levels (p = 0.94) and stationarity of log returns (p < 0.001).
   - Fits ARIMA(1,0,1) models to daily returns: AR(1) and MA(1) coefficients are statistically insignificant (p > 0.05).
   - Confirms the Efficient Market Hypothesis (EMH) / near-random walk at the daily horizon.
   - **Theoretical conclusion**: Predicting price directions is near impossible on daily horizons; however, volatility exhibits clustering (the ARCH effect), motivating volatility regime modeling.

2. **`02_sector_momentum_preview.ipynb` — Cross-Sectional Momentum Signals**
   - Composite momentum ranking across 10 major Nifty sector indices (Auto, Bank, FMCG, IT, Metal, Pharma, PSU Bank, Realty, Financial Services, Energy).
   - Demonstrates that while time-series returns of broad market indices are near-random walk, cross-sectional relative strength momentum persists across sectors.
   - Serves as the natural companion piece for sector rotation allocation under varying volatility regimes.
