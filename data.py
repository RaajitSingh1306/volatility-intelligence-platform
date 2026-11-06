"""
data.py
-------
Fetch Nifty 50 OHLCV data from Yahoo Finance and compute return series.

Usage
-----
    from data import fetch_nifty
    df = fetch_nifty()
"""

import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "nifty_ohlcv.csv"
TICKER = "^NSEI"
CACHE_MAX_AGE_DAYS = 7

def fetch_nifty(
    start: str = "2014-01-01",
    end: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download or load cached Nifty 50 OHLCV data.

    Parameters
    ----------
    start : str
        Start date in YYYY-MM-DD format.
    end : str | None
        End date in YYYY-MM-DD format.
        If None, today's date is used.
    cache : bool
        If True and a cached CSV exists and is less than 7 days old,
        load from disk instead of downloading.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        Open, High, Low, Close, Volume, returns, log_returns.
        Index is a DatetimeIndex.

    Raises
    ------
    ValueError
        If the downloaded data is empty.
    """

    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Use cache if it exists and is less than 7 days old
    if cache and DATA_PATH.exists():
        age_days = (time.time() - DATA_PATH.stat().st_mtime) / 86400

        if age_days < CACHE_MAX_AGE_DAYS:
            logger.info(
                "Loading cached data (%d days old)...",
                int(age_days),
            )
            return pd.read_csv(
                DATA_PATH,
                index_col=0,
                parse_dates=True,
            )

        logger.info(
            "Cache stale (%d days old), re-downloading...",
            int(age_days),
        )

    logger.info(
        "Downloading %s from Yahoo Finance (%s → %s)...",
        TICKER,
        start,
        end,
    )

    raw: pd.DataFrame = yf.download(
        TICKER,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"No data returned for {TICKER}. "
            "Check ticker and date range."
        )

    # Flatten MultiIndex columns produced by some yfinance versions
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[
        ["Open", "High", "Low", "Close", "Volume"]
    ].copy()

    df.dropna(inplace=True)

    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    df.dropna(inplace=True)

    df.to_csv(DATA_PATH)

    logger.info(
        "Saved %d rows → %s",
        len(df),
        DATA_PATH,
    )

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    _df = fetch_nifty()

    print(_df.tail())
    print(_df.dtypes)