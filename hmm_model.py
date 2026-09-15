"""
hmm_model.py
------------
Hidden Markov Model (HMM) regime labeling + SHAP explainability.

Pipeline
--------
1. Standardize the 15 engineered features.
2. Fit a 3-state Gaussian HMM.
3. Remap raw HMM states to Low / Mid / High Vol by mean GARCH conditional vol.
4. Train a Gradient Boosting surrogate to expose feature importance via SHAP.

Regime mapping
--------------
    0  →  Low Vol   (green)
    1  →  Mid Vol   (orange)
    2  →  High Vol  (red)

Usage
-----
    from hmm_model import fit_hmm, plot_regimes, shap_explain
    df, hmm_model, scaler = fit_hmm(df)
    plot_regimes(df)
    shap_explain(df)
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")

N_REGIMES = 3
REGIME_NAMES: dict[int, str] = {0: "Low Vol", 1: "Mid Vol", 2: "High Vol"}
REGIME_COLORS: dict[int, str] = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}


# ── HMM ──────────────────────────────────────────────────────────────────────

def fit_hmm(
    df: pd.DataFrame,
    n_states: int = N_REGIMES,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, GaussianHMM, StandardScaler]:
    """
    Fit a Gaussian HMM and attach regime labels to *df*.

    States are ordered by ascending mean GARCH volatility so that regime 0 is
    always the lowest-volatility state regardless of HMM initialisation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all columns in ``FEATURE_COLS`` and ``garch_vol``.
    n_states : int
        Number of HMM hidden states.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, GaussianHMM, StandardScaler]
        - df with new columns: raw_regime, regime (int), regime_name (str)
        - fitted GaussianHMM
        - fitted StandardScaler (use for inference on new data)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
    )
    model.fit(X)
    log_score = model.score(X)
    logger.info("HMM log-likelihood: %.4f", log_score)

    raw_labels = model.predict(X)
    out = df.copy()
    out["raw_regime"] = raw_labels

    # Order states so 0 = lowest volatility
    mean_vol = out.groupby("raw_regime")["garch_vol"].mean().sort_values()
    remap = {old: new for new, old in enumerate(mean_vol.index)}
    out["regime"] = out["raw_regime"].map(remap)
    out["regime_name"] = out["regime"].map(REGIME_NAMES)

    _print_regime_stats(out)
    return out, model, scaler


def _print_regime_stats(df: pd.DataFrame) -> None:
    stats = (
        df.groupby("regime_name")
        .agg(
            days=("regime", "count"),
            mean_vol=("garch_vol", "mean"),
            mean_ret=("returns", "mean"),
            ret_std=("returns", "std"),
        )
        .round(4)
    )
    logger.info("\n── Regime Statistics ──\n%s", stats.to_string())


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_regimes(
    df: pd.DataFrame,
    save_path: str = "outputs/regimes.png",
) -> None:
    """
    Two-panel chart: price with regime shading, and GARCH vol coloured by regime.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: Close, regime, regime_name, garch_vol.
    save_path : str
        Output file path.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(
        "HMM Volatility Regime Classification — Nifty 50",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(df.index, df["Close"], color="black", linewidth=0.7, zorder=5)
    ax.set_ylabel("Nifty 50 Price (₹)")
    price_min, price_max = df["Close"].min(), df["Close"].max()
    for regime, color in REGIME_COLORS.items():
        mask = df["regime"] == regime
        ax.fill_between(df.index, price_min, price_max, where=mask, alpha=0.25, color=color)
    patches = [
        mpatches.Patch(color=c, label=REGIME_NAMES[r], alpha=0.6)
        for r, c in REGIME_COLORS.items()
    ]
    ax.legend(handles=patches, loc="upper left")

    ax2 = axes[1]
    for regime, color in REGIME_COLORS.items():
        mask = df["regime"] == regime
        ax2.fill_between(
            df.index,
            0,
            df["garch_vol"] * 100,
            where=mask,
            color=color,
            alpha=0.6,
            label=REGIME_NAMES[regime],
        )
    ax2.set_ylabel("Ann. GARCH Vol (%)")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", save_path)


# ── SHAP Explainability ───────────────────────────────────────────────────────

def shap_explain(
    df: pd.DataFrame,
    save_path: str = "outputs/shap_summary.png",
) -> pd.Series:
    """
    Train a GBM surrogate on HMM regime labels and explain via SHAP.

    The GBM is a post-hoc interpretability tool — it reveals which engineered
    features drive the HMM regime classifications.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all ``FEATURE_COLS`` and a ``regime`` column.
    save_path : str
        Output file path for SHAP bar chart.

    Returns
    -------
    pd.Series
        Mean absolute SHAP importance per feature, sorted descending.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLS]
    y = df["regime"]

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    clf.fit(X, y)
    train_acc = (clf.predict(X) == y).mean()
    logger.info("GBM surrogate train accuracy: %.3f", train_acc)

    background = shap.sample(X, min(len(X), 100), random_state=42)
    explain_X = shap.sample(X, min(len(X), 250), random_state=42)
    explainer = shap.Explainer(
        clf.predict_proba,
        background,
        algorithm="permutation",
    )
    shap_values = explainer(explain_X).values  # (samples, features, classes)

    mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    importance = pd.Series(mean_abs, index=FEATURE_COLS).sort_values(ascending=False)
    logger.info("\n── SHAP Feature Importance ──\n%s", importance.round(4).to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.sort_values().plot.barh(ax=ax, color="#5C6BC0")
    ax.set_title(
        "SHAP Feature Importance — Volatility Regime Classifier",
        fontweight="bold",
    )
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", save_path)

    return importance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from data import fetch_nifty
    from features import build_features
    from garch_model import add_garch_vol

    _df = fetch_nifty()
    _df = build_features(_df)
    _df, _ = add_garch_vol(_df)
    _df, _model, _scaler = fit_hmm(_df)
    plot_regimes(_df)
    shap_explain(_df)
    _df.to_csv("data/labeled_data.csv")
    logger.info("Labeled data saved → data/labeled_data.csv")