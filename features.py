"""
features.py
-----------
Volatility and momentum feature engineering for Nifty 50 OHLCV data.

Features produced
-----------------
- Realized volatility: rv_5d, rv_10d, rv_21d, rv_63d
- Vol-of-vol ratios: vol_ratio_5_21, vol_ratio_21_63
- OHLC vol estimators: gk_vol_21 (Garman-Klass), park_vol_21 (Parkinson)
- Price momentum: mom_5d, mom_10d, mom_21d, mom_63d
- RSI (14-period)
- Relative volume: vol_ma_ratio
- Rolling drawdown: drawdown

Usage
-----
    from features import build_features
    df = build_features(df)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

FEATURE_COLS: list[str] = [
    "rv_5d", "rv_10d", "rv_21d", "rv_63d",
    "vol_ratio_5_21", "vol_ratio_21_63",
    "gk_vol_21", "park_vol_21",
    "mom_5d", "mom_10d", "mom_21d", "mom_63d",
    "rsi_14", "vol_ma_ratio", "drawdown",
]


def garman_klass_vol(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Annualized Garman-Klass volatility estimator (OHLC-based).

    More statistically efficient than close-to-close estimator.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: High, Low, Open, Close.
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.Series
        Annualized Garman-Klass volatility.
    """
    log_hl = np.log(df["High"] / df["Low"]) ** 2
    log_co = np.log(df["Close"] / df["Open"]) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return gk.rolling(window).mean().apply(np.sqrt) * np.sqrt(TRADING_DAYS)


def parkinson_vol(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Annualized Parkinson volatility estimator (High-Low based).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: High, Low.
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.Series
        Annualized Parkinson volatility.
    """
    log_hl = np.log(df["High"] / df["Low"]) ** 2
    factor = 1.0 / (4.0 * np.log(2))
    return (factor * log_hl).rolling(window).mean().apply(np.sqrt) * np.sqrt(TRADING_DAYS)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all features from an OHLCV + returns DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: Open, High, Low, Close, Volume, log_returns, returns.

    Returns
    -------
    pd.DataFrame
        Original columns plus all engineered features. Rows with NaN dropped.
    """
    feat = df.copy()

    # ── Realized Volatility ──────────────────────────────────────────────────
    for w in [5, 10, 21, 63]:
        feat[f"rv_{w}d"] = feat["log_returns"].rolling(w).std() * np.sqrt(TRADING_DAYS)

    # ── Vol Ratios ───────────────────────────────────────────────────────────
    feat["vol_ratio_5_21"] = feat["rv_5d"] / feat["rv_21d"]
    feat["vol_ratio_21_63"] = feat["rv_21d"] / feat["rv_63d"]

    # ── OHLC Estimators ──────────────────────────────────────────────────────
    feat["gk_vol_21"] = garman_klass_vol(df, 21)
    feat["park_vol_21"] = parkinson_vol(df, 21)

    # ── Momentum ─────────────────────────────────────────────────────────────
    for w in [5, 10, 21, 63]:
        feat[f"mom_{w}d"] = feat["Close"].pct_change(w)

    # ── RSI ──────────────────────────────────────────────────────────────────
    feat["rsi_14"] = _rsi(feat["Close"], 14)

    # ── Relative Volume ──────────────────────────────────────────────────────
    feat["vol_ma_ratio"] = feat["Volume"] / feat["Volume"].rolling(21).mean()

    # ── Rolling Drawdown from Peak ───────────────────────────────────────────
    roll_max = feat["Close"].cummax()
    feat["drawdown"] = (feat["Close"] - roll_max) / roll_max

    before = len(feat)
    feat.dropna(inplace=True)
    logger.info("Features built: %d → %d rows after dropna", before, len(feat))
    return feat


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from data import fetch_nifty

    _df = fetch_nifty()
    _feat = build_features(_df)
    print(_feat[FEATURE_COLS].describe().round(4))