"""
run.py
------
Master pipeline: data → features → GARCH → HMM → backtest.

Run the full pipeline end-to-end:
    python run.py

Individual modules can also be run standalone — see their ``if __name__ == "__main__"`` blocks.
"""

import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

Path("data").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)


def main() -> None:
    logger.info("=" * 55)
    logger.info("  Volatility Regime Classifier — Full Pipeline")
    logger.info("=" * 55)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    logger.info("[1/5] Fetching Nifty 50 data...")
    from data import fetch_nifty
    df = fetch_nifty()

    # ── 2. Features ──────────────────────────────────────────────────────────
    logger.info("[2/5] Engineering features...")
    from features import build_features
    df = build_features(df)

    # ── 3. GARCH ─────────────────────────────────────────────────────────────
    logger.info("[3/5] Fitting GARCH(1,1)...")
    from garch_model import add_garch_vol, plot_garch_vol
    df, garch = add_garch_vol(df)
    plot_garch_vol(df)

    # ── 4. HMM + SHAP ────────────────────────────────────────────────────────
    logger.info("[4/5] Running HMM regime labeling + SHAP explainability...")
    from hmm_model import fit_hmm, plot_regimes, shap_explain
    df, hmm, scaler = fit_hmm(df)
    plot_regimes(df)
    shap_explain(df)

    # ── Save labeled dataset ─────────────────────────────────────────────────
    out_path = "data/labeled_data.csv"
    df.to_csv(out_path)
    logger.info("Labeled dataset saved → %s", out_path)

    # ── 5. Backtest ──────────────────────────────────────────────────────────
    logger.info("[5/5] Running vectorbt regime-based backtest...")
    from backtest import run_backtest
    summary, pf = run_backtest(df)
    summary.to_csv("data/backtest_summary.csv")
    logger.info("Backtest summary saved → data/backtest_summary.csv")

    logger.info("=" * 55)
    logger.info("  Pipeline complete!")
    logger.info("  Dashboard: streamlit run app/dashboard.py")
    logger.info("  API:       uvicorn app.main:app --reload")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()