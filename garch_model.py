"""
garch_model.py
--------------
Fit a GARCH(1,1) model on Nifty 50 log-returns and extract conditional volatility.

The GARCH(1,1) model is:
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Conditional volatility is de-scaled and annualized before being attached to the
feature DataFrame.

Usage
-----
    from garch_model import add_garch_vol, plot_garch_vol
    df, garch_result = add_garch_vol(df)
    plot_garch_vol(df)
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")
TRADING_DAYS = 252
SCALE_FACTOR = 100  # numeric stability for GARCH fitting


def fit_garch(
    log_returns: pd.Series,
    p: int = 1,
    q: int = 1,
) -> dict:
    """
    Fit a GARCH(p,q) model with normal innovations on log-returns.

    Parameters
    ----------
    log_returns : pd.Series
        Daily log-return series.
    p : int
        GARCH lag order for squared errors.
    q : int
        GARCH lag order for conditional variance.

    Returns
    -------
    dict with keys:
        - ``result``   : ARCHModelResult
        - ``cond_vol`` : pd.Series — annualized conditional volatility
        - ``aic``      : float
        - ``bic``      : float
    """
    scaled = log_returns * SCALE_FACTOR
    model = arch_model(scaled, vol="Garch", p=p, q=q, dist="normal")
    result: ARCHModelResult = model.fit(disp="off")

    cond_vol = result.conditional_volatility / SCALE_FACTOR * np.sqrt(TRADING_DAYS)
    cond_vol.index = log_returns.index

    logger.info("GARCH(%d,%d) — AIC: %.2f  BIC: %.2f", p, q, result.aic, result.bic)
    return {
        "result": result,
        "cond_vol": cond_vol,
        "aic": result.aic,
        "bic": result.bic,
    }


def add_garch_vol(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Fit GARCH(1,1) and append ``garch_vol`` column (annualized) to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``log_returns`` column.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Updated DataFrame and GARCH result dict.
    """
    garch = fit_garch(df["log_returns"])
    out = df.copy()
    out["garch_vol"] = garch["cond_vol"].reindex(out.index)
    return out, garch


def plot_garch_vol(
    df: pd.DataFrame,
    save_path: str = "outputs/garch_vol.png",
) -> None:
    """
    Plot Nifty 50 price, daily log-returns, and GARCH conditional volatility.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Close, log_returns, garch_vol.
    save_path : str
        File path for the saved figure.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        "GARCH(1,1) Conditional Volatility — Nifty 50",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].plot(df["Close"], color="#2196F3", linewidth=0.8)
    axes[0].set_ylabel("Price (₹)")
    axes[0].set_title("Nifty 50 Close")

    axes[1].plot(df["log_returns"] * 100, color="#9E9E9E", linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel("Log Return (%)")
    axes[1].set_title("Daily Log Returns")

    axes[2].plot(df["garch_vol"] * 100, color="#E91E63", linewidth=0.9)
    axes[2].set_ylabel("Ann. Vol (%)")
    axes[2].set_title("GARCH(1,1) Conditional Volatility (Annualized)")
    axes[2].axhline(15, color="green", linestyle="--", linewidth=0.8, label="Low threshold (15%)")
    axes[2].axhline(30, color="orange", linestyle="--", linewidth=0.8, label="High threshold (30%)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", save_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from data import fetch_nifty
    from features import build_features

    _df = fetch_nifty()
    _df = build_features(_df)
    _df, _garch = add_garch_vol(_df)
    plot_garch_vol(_df)
    print(_df[["garch_vol"]].describe())