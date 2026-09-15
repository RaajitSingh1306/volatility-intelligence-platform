"""
backtest.py
-----------
Vectorbt-powered regime-based strategy backtest.

Strategy logic
--------------
+----------+--------+------------------------------------------+
| Regime   | Signal | Rationale                                |
+----------+--------+------------------------------------------+
| Low Vol  | ENTRY  | Trending market, low uncertainty         |
| Mid Vol  | HOLD   | Stay in existing position                |
| High Vol | EXIT   | Protect capital, reduce drawdown risk    |
+----------+--------+------------------------------------------+

Benchmark: Buy & Hold with identical capital and no fees.

Usage
-----
    from backtest import run_backtest
    summary, pf = run_backtest(df)
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")
INIT_CASH = 100_000
FEES = 0.001      # 0.1 % per trade
SLIPPAGE = 0.001


def run_backtest(
    df: pd.DataFrame,
    save_path: str = "outputs/backtest.png",
) -> Tuple[pd.DataFrame, "vbt.Portfolio"]:
    """
    Run regime-based strategy against Buy & Hold and return performance summary.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Close, regime.
        ``regime == 0`` → Low Vol (entry signal)
        ``regime == 2`` → High Vol (exit signal)
    save_path : str
        Output file path for the equity/drawdown chart.

    Returns
    -------
    Tuple[pd.DataFrame, vbt.Portfolio]
        - summary DataFrame with strategy vs benchmark metrics
        - the vectorbt Portfolio object for further analysis
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    price = df["Close"]
    entries = df["regime"] == 0  # enter on Low Vol
    exits = df["regime"] == 2    # exit on High Vol

    pf = vbt.Portfolio.from_signals(
        close=price,
        entries=entries,
        exits=exits,
        init_cash=INIT_CASH,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="D",
    )

    bh = vbt.Portfolio.from_holding(price, init_cash=INIT_CASH, freq="D")

    summary = pd.DataFrame(
        {"Strategy": _collect_metrics(pf), "Buy & Hold": _collect_metrics(bh)}
    ).T

    logger.info("\n── Backtest Results ──\n%s", summary.to_string())

    _plot(pf, bh, save_path)
    return summary, pf


def _collect_metrics(pf: "vbt.Portfolio") -> dict:
    """Extract key performance metrics from a vectorbt Portfolio."""
    has_trades = len(pf.trades.records) > 0
    return {
        "Total Return (%)": round(pf.total_return() * 100, 2),
        "CAGR (%)": round(pf.annualized_return() * 100, 2),
        "Sharpe Ratio": round(pf.sharpe_ratio(), 3),
        "Max Drawdown (%)": round(pf.max_drawdown() * 100, 2),
        "Calmar Ratio": round(pf.calmar_ratio(), 3),
        "Win Rate (%)": round(pf.trades.win_rate() * 100, 2) if has_trades else float("nan"),
        "Num Trades": len(pf.trades.records),
    }


def _plot(
    pf: "vbt.Portfolio",
    bh: "vbt.Portfolio",
    save_path: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "Volatility Regime Strategy vs Buy & Hold",
        fontsize=13,
        fontweight="bold",
    )

    pf.value().rename("Strategy").plot(ax=axes[0], color="#2196F3")
    bh.value().rename("Buy & Hold").plot(ax=axes[0], color="#9E9E9E", linestyle="--")
    axes[0].set_ylabel("Portfolio Value (₹)")
    axes[0].legend()

    pf.drawdown().rename("Strategy DD").plot(ax=axes[1], color="#F44336")
    bh.drawdown().rename("B&H DD").plot(ax=axes[1], color="#9E9E9E", linestyle="--")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", save_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    _df = pd.read_csv("data/labeled_data.csv", index_col=0, parse_dates=True)
    _summary, _pf = run_backtest(_df)
    _summary.to_csv("data/backtest_summary.csv")