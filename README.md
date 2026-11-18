# Volatility Intelligence Platform — Nifty 50 (v3 Flagship)

> [!NOTE]
> **FINAL SHIPPED PRODUCTION PLATFORM**  
> This system is the **final, shipped production version** of the Volatility Regime classification platform. It officially **supersedes** both earlier developmental iterations:
> - **[Volatility Regime Classifier (v1 Prototype)](https://github.com/RaajitSingh1306/Volatility-Regime-Classifier)** (dual-asset prototype)
> - **[Volatility Classifier Simplified (v2 Refactor)](https://github.com/RaajitSingh1306/volatility-classifier-simplified)** (single-asset modular refactor)

[![Status: Production Flagship](https://img.shields.io/badge/Status-PRODUCTION%20FLAGSHIP-emerald)](#evolution--supersession-lineage)
[![CI Tests](https://img.shields.io/badge/tests-24%2F24%20passed-brightgreen)](#8-results--evaluation)
[![OOF Macro-AUC](https://img.shields.io/badge/OOF%20Macro--AUC-0.6515-orange)](#8-results--evaluation)
[![Sharpe Ratio](https://img.shields.io/badge/Sharpe-0.269%20vs%200.305%20B%26H-orange)](#8-results--evaluation)
[![Docker](https://img.shields.io/badge/Docker-Dockerfile%20included-lightgrey)](#12-deployment)
[![Deploy on Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7)](#12-deployment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#16-license--disclaimer)

| Resource | URL |
|---|---|
| **Web Dashboard** | [https://volatility-intelligence-platform.vercel.app](https://volatility-intelligence-platform.vercel.app) |
| **REST API** | [https://volatility-intelligence-platform.onrender.com](https://volatility-intelligence-platform.onrender.com) |
| **Interactive API Docs** | [https://volatility-intelligence-platform.onrender.com/docs](https://volatility-intelligence-platform.onrender.com/docs) |

A quantitative research platform that classifies and backtests **Nifty 50 (`^NSEI`)** volatility regimes: **GARCH(1,1)** conditional volatility, a **3-state Gaussian HMM** for regime labels, a **5-fold walk-forward XGBoost ensemble** that predicts the next-day label (pooled OOF Macro-AUC 0.6515 on 51 Mid/High days; a "tomorrow = today" baseline scores 0.9455), **TreeSHAP** attributions, **MLflow** tracking and a **vectorbt** backtest (1-day lag, 0.1% fees, 0.1% slippage), served via **FastAPI** and a **Next.js 14** console.

---

## Evolution & Supersession Lineage

This platform represents the culmination of three developmental generations:

| Dimension | v1: Volatility Classifier (Prototype) | v2: Volatility Classifier Simplified | v3: Volatility Intelligence Platform (This System) |
|---|---|---|---|
| **Scope** | Nifty 50 + Bank Nifty | Nifty 50 single-asset | **Nifty 50 full production platform** |
| **Features** | 3 engineered volatility features | 15 volatility & momentum features | **15 standardized econometric & technical features** |
| **Model Architecture** | Gaussian HMM + GARCH(1,1) | Gaussian HMM + GARCH(1,1) | **HMM state clustering + 5-fold XGBoost forward predictive ensemble** |
| **Prediction Horizon** | Retrospective state discovery | Retrospective state discovery | **Day $T \rightarrow$ Day $T+1$ prospective probability distribution** |
| **Evaluation** | Descriptive confusion matrix | Basic backtest metrics | **Walk-forward CV with 5-day gap (pooled OOF Macro-AUC 0.6515; folds 1 and 4 undefined)** |
| **Explainability** | Kernel SHAP ($O(2^N)$ approximation) | Gradient Boosting surrogate SHAP | **TreeExplainer exact Shapley attribution ($O(TLD^2)$ polynomial)** |
| **Experiment Tracking** | None | None | **MLflow integration (`mlflow.db` SQLite, 3-run sweep)** |
| **Backtesting Engine** | Custom pandas script | vectorbt backtest | **vectorbt realistic backtest (1-day execution lag, 0.1% fees, 0.1% slippage)** |
| **Serving API** | FastAPI (4 endpoints) | FastAPI (5 endpoints) | **FastAPI (7 endpoints; data and models cached with `@lru_cache`; latency not benchmarked)** |
| **Frontend UI** | Streamlit | Streamlit | **Next.js 14 + Tailwind CSS dark web console** |
| **Test Coverage** | Manual | 13 Pytest unit tests | **24 pytest tests (features, GARCH, HMM, backtest, models, API)** |
| **Containerization** | None | Basic Dockerfile | **Single-stage Dockerfile; models trained during `docker build`** |
| **Status** | *Superseded* | *Superseded* | **Active Flagship Shipped Platform** |

---

## Table of Contents

- [1. What This Project Does](#1-what-this-project-does)
- [2. Why It Was Built](#2-why-it-was-built)
- [3. System Architecture](#3-system-architecture)
- [4. Tech Stack & Libraries](#4-tech-stack--libraries)
- [5. Data](#5-data)
- [6. Step-by-Step Pipeline](#6-step-by-step-pipeline)
- [7. Problems Faced & How We Solved Them](#7-problems-faced--how-we-solved-them)
- [8. Results & Evaluation](#8-results--evaluation)
- [9. Project Structure](#9-project-structure)
- [10. Getting Started](#10-getting-started)
- [11. API Reference](#11-api-reference)
- [12. Deployment](#12-deployment)
- [13. Connected Portfolio Projects](#13-connected-portfolio-projects)
- [14. Limitations & Known Issues](#14-limitations--known-issues)
- [15. Roadmap / Future Expansion](#15-roadmap--future-expansion)
- [16. License & Disclaimer](#16-license--disclaimer)

---

## 1. What This Project Does

The Volatility Intelligence Platform operates on daily Nifty 50 market data to provide:

- **Unsupervised Regime Discovery**: Identifies latent market states via a 3-state Gaussian HMM parameterized across 15 features without arbitrary manual thresholds.
- **Forward-Looking Regime Forecasting**: Uses an expanding-window 5-fold XGBoost ensemble to predict tomorrow's regime ($P(z_{T+1} = k \mid x_T)$). Pooled out-of-fold Macro-AUC is 0.6515, computed on only 51 Mid/High days; a naive "tomorrow = today" baseline scores 0.9455.
- **Continuous Econometric Volatility**: Fits daily GARCH(1,1) conditional volatility to capture volatility clustering.
- **TreeSHAP Explainability**: Computes exact Shapley attributions for all 15 features on the last-fold XGBoost model over the most recent 500 rows (`scripts/regime_classifier.py`).
- **Regime-Switching Backtest**: Enters on Low Vol, holds through Mid Vol and exits to cash on High Vol. Signals are shifted one day; costs are 0.1% fees and 0.1% slippage. Over 2014-04-07 to 2026-09-11 it earned less than buy-and-hold (Section 8).
- **Dual Cloud Serving**: Deploys a containerized FastAPI backend on Render and a responsive dark-mode web console on Vercel.

### Regime Definitions & Systematic Actions

| Regime | Label | Market Environment | Characteristic Behavior | Strategy Position |
|---|:---:|---|---|:---:|
| **Low Vol** | `0` | Normal conditions: 97.3% of days (2,971), includes stress up to ~48% GARCH vol | Default state | **Enter / stay long** |
| **Mid Vol** | `1` | Short episodes: 57 days in 7 runs, longest 28 days | Scattered, no single market story | **Hold current position** |
| **High Vol** | `2` | One episode: 2020-03-13 to 2020-04-24 (27 days, 0.9%) | COVID crash | **Exit to cash** |

Source: `results/results.json`. The HMM mainly isolates the COVID crash; it does not produce three balanced regimes.

---

## 2. Why It Was Built

- **Replacing Market Intuition with Statistical Rigor**: Retail investors and portfolio managers often rely on qualitative sentiment to gauge market risk. This leads to late exits during rising volatility and delayed entries during market recovery rallies.
- **Eliminating Backward-Looking Model Traps**: Traditional GARCH and HMM tell you what happened up to today; they do not provide a prospective forecast for tomorrow. VIP bridges unsupervised regime discovery with supervised forward-looking XGBoost classification.
- **Preventing Time-Series Data Leakage**: Standard shuffled cross-validation or overlapping rolling windows leak future information into validation folds. VIP uses an expanding-window walk-forward split with a **5-day gap**. It removes short-window leakage but only partly covers the 21- and 63-day features.
- **Reproducible Research**: Every number in Section 8 comes from one pinned run saved to `results/`, with experiment tracking via MLflow.

---

## 3. System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. DATA INGESTION LAYER                            │
│                                (data.py)                                    │
│   Yahoo Finance (^NSEI, 2014 to present) ──► data/nifty_ohlcv.csv (7-day TTL)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ OHLCV: Date, Open, High, Low, Close, Vol
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       2. FEATURE ENGINEERING LAYER                          │
│                               (features.py)                                 │
│   15 Engineered Features across 5 Categories:                               │
│   Realized vol · Vol-of-vol · Range estimators · Momentum · Microstructure  │
└──────────────────┬─────────────────────────────────────┬────────────────────┘
                   │                                     │
                   ▼                                     ▼
┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
│        GARCH(1,1) ECONOMETRIC        │  │       3-STATE GAUSSIAN HMM        │
│           (garch_model.py)           │  │          (hmm_model.py)           │
│                                      │  │                                   │
│   σ²ₜ = ω + αε²ₜ₋₁ + βσ²ₜ₋₁          │  │   Latent state clustering         │
│   → garch_vol series                 │  │   Sorted: 0=Low, 1=Mid, 2=High    │
└──────────────────┬───────────────────┘  └──────────────────┬────────────────┘
                   │                                         │
                   └───────────────────┬─────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   3. XGBOOST FORWARD PREDICTIVE ENSEMBLE                    │
│                        (scripts/regime_classifier.py)                       │
│                                                                             │
│   Input:   X = 15 features on day T                                         │
│   Target:  y = HMM regime label on day T+1 (TOMORROW)                       │
│   CV:      TimeSeriesSplit(n_splits=5, gap=5)                               │
│   Model:   XGBClassifier(multi:softprob, 3 classes, 5-fold ensemble)        │
│   Explain: TreeExplainer SHAP (exact Shapley values)                        │
│   Artifacts: models/xgb_bundle.pkl · outputs/shap_summary_xgb.png           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
│        4. MLFLOW EXPERIMENT DB       │  │        5. VECTORBT BACKTEST       │
│      (scripts/mlflow_tracking.py)    │  │          (backtest.py)            │
│   mlflow.db (SQLite backend)         │  │                                   │
│   3 sweeps: n_est ∈ {200, 300, 500}  │  │   Enter Low, hold Mid,            │
│   Params, fold AUCs, OOF AUC logged  │  │   Exit to cash on High            │
│   → mlruns/ · mlflow.db              │  │   Friction: 1-day lag, 0.1% fees  │
└──────────────────────────────────────┘  └──────────────────┬────────────────┘
                                                             │
                                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            6. SERVING LAYER                                 │
│                                                                             │
│   FastAPI Backend (app/main.py)           Next.js Frontend (frontend/)      │
│   Render / Docker :8000                   Vercel :3000                      │
│                                                                             │
│   GET /current                            Hero KPI summary cards            │
│   GET /predict                            Regime probability distribution   │
│   GET /market-summary                     SHAP bars (top 7 from API)        │
│   GET /regimes?last_n=252                 Regime history timeline feed      │
│   GET /stats                              Per-regime statistics breakdown   │
│   GET /backtest-summary                   Strategy equity curve vs B&H      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tech Stack & Libraries

| Library / Tool | Version | Purpose | Rationale |
|---|---|---|---|
| **Python** | `3.12` (reported run: 3.12.10) | Core programming runtime | Modern high-performance scientific programming |
| **XGBoost** | `3.3.0` | Forward regime classification | Gradient boosted decision trees for forward-looking $P(z_{T+1} \mid x_T)$ predictions |
| **hmmlearn** | `0.3.3` | Unsupervised regime discovery | Gaussian Hidden Markov Models with full covariance for multivariate state discovery |
| **arch** | `8.0.0` | Econometric volatility modeling | Industry-standard conditional heteroskedasticity GARCH(1,1) estimation |
| **scikit-learn** | `1.9.0` | Preprocessing & validation | `StandardScaler`, `TimeSeriesSplit(gap=5)`, and multi-class ROC-AUC metrics |
| **SHAP** | `0.52.0` | Model explainability | `TreeExplainer` providing exact polynomial-time Shapley attributions |
| **MLflow** | `3.15.1` | Experiment tracking | SQLite-backed registry tracking hyperparameters, fold AUCs, and model artifacts |
| **vectorbt** | `1.1.0` | Vectorized backtesting | Signal-based simulation with 0.1% fees, 0.1% slippage and a 1-day signal lag |
| **FastAPI** | `0.141.1` | Production REST API | Asynchronous endpoints with Pydantic validation and in-memory `@lru_cache` |
| **Next.js** | `^14.2.21` | Web console | Modern React 18 framework with SSR, static optimization, and dark-mode styling |
| **Tailwind CSS** | `^3.4.16` | Frontend styling | Utility-first CSS framework with tailored slate-and-emerald dark tokens |
| **Docker** | `python:3.12-slim`, single `FROM` | Containerization | Models are trained during `docker build`, so the container does not train at startup |
| **pytest** | `8.3.3` | Automated CI test suite | 24 unit and integration tests across features, GARCH, HMM, backtest, models, and API |

---

## 5. Data

- **Asset**: Nifty 50 Index (`^NSEI`).
- **Source**: Yahoo Finance API via `yfinance`.
- **Sample Horizon**: 2014-01-01 to 2026-09-12 requested; the reported run has 3,055 rows (2014-04-07 to 2026-09-11) after feature warm-up. Spans 2015-16 EM selloff, 2018 NBFC crisis, March 2020 COVID shock, and 2021-2024 expansion.
- **Cache**: `data/nifty_ohlcv.csv` with a **7-day TTL** for rapid startup.
- **15 Quantitative Features**:

| Category | Features | Formulation / Description |
|---|---|---|
| **Realized Volatility (4)** | `rv_5d`, `rv_10d`, `rv_21d`, `rv_63d` | Rolling standard deviation of continuous log returns scaled by $\sqrt{252}$ |
| **Vol-of-Vol Ratios (2)** | `vol_ratio_5_21`, `vol_ratio_21_63` | Short-term to medium/long-term volatility ratios ($>1.0$ indicates acceleration) |
| **OHLC Estimators (2)** | `park_vol_21`, `gk_vol_21` | Parkinson ($\sim 2.5\times$ efficiency) and Garman-Klass ($\sim 5\times$ efficiency over close-to-close) |
| **Momentum (4)** | `mom_5d`, `mom_10d`, `mom_21d`, `mom_63d` | Multi-horizon price percentage changes $(P_t - P_{t-k}) / P_{t-k}$ |
| **Microstructure & Risk (3)** | `rsi_14`, `vol_ma_ratio`, `drawdown` | 14-day RSI (simple rolling mean of gains and losses, not Wilder smoothing), volume divided by its 21-day mean, and high-water mark peak-to-trough drawdown |

---

## 6. Step-by-Step Pipeline

1. **Market Data Retrieval (`data.py`)**: Download Nifty 50 daily OHLCV from Yahoo Finance; store to local cache if older than 7 days; calculate simple and log returns.
2. **Feature Engineering (`features.py`)**: Compute the 15 quantitative features across realized volatility, Parkinson/Garman-Klass estimators, momentum, and RSI.
3. **GARCH(1,1) Volatility Modeling (`garch_model.py`)**: Scale returns by $\times 100$ and fit constant-mean GARCH(1,1):
   $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
4. **Gaussian HMM Regime Discovery (`hmm_model.py`)**: Standardize the 15-feature matrix; fit 3-state Gaussian HMM with full covariance; sort states ascending by mean GARCH volatility so that $0 = \text{Low}$, $1 = \text{Mid}$, $2 = \text{High}$.
5. **XGBoost Forward Predictive Modeling (`scripts/regime_classifier.py`)**:
   - Construct targets: $y_t = z_{t+1}$ (predicting tomorrow's HMM state from today's features).
   - Evaluate via 5-fold walk-forward cross-validation with a strict 5-day gap (`TimeSeriesSplit(n_splits=5, gap=5)`).
   - Train multi-class softprob ensemble across all 5 folds; log pooled OOF Macro-AUC (0.6515, best of the 3 sweep settings).
6. **MLflow Experiment Tracking (`scripts/mlflow_tracking.py`)**: Log hyperparameters, fold AUCs, and model artifacts to `mlflow.db`.
7. **TreeSHAP Feature Attribution**: Compute exact Shapley attributions with `shap.TreeExplainer` on the last walk-forward fold model, over the latest 500 rows.
8. **Vectorized Backtesting (`backtest.py`)**: Simulate the regime strategy using `vectorbt` with 0.1% fees, 0.1% slippage, and 1-day execution lag.
9. **Serving & Dashboard Delivery**: Launch FastAPI REST server (`app/main.py`) with `@lru_cache` and connect the Next.js 14 web console.

---

## 7. Problems Faced & How We Solved Them

| Problem | Impact | How We Got Around It |
|---|---|---|
| **Walk-forward CV data leakage from rolling indicators** | 5-day rolling features (`rv_5d`, `mom_5d`) and GARCH variance memory can leak information across train/validation boundaries | Used `TimeSeriesSplit(n_splits=5, gap=5)` so a 5-day gap separates training and validation windows. Longer windows (`rv_63d`, `mom_63d`) are only partly covered by this gap. |
| **Early stopping on the validation fold** | Early stopping that watches the validation fold picks the stopping point on the data it is then scored on, which inflates AUC | Early stopping now uses the last 15% of each training window; the validation fold is used only for scoring |
| **GARCH numerical conditioning failure** | Raw daily returns (~0.001) produced variances near machine epsilon, causing Maximum Likelihood Optimization (MLE) gradients to fail to converge | Scaled returns by **$\times 100$ before GARCH estimation** (`returns * 100`), keeping gradient norms well above machine epsilon. Rescaled fitted conditional variance back to annualized percentage volatility |
| **HMM state permutation** (inherited from v1/v2) | Random state indices after EM convergence caused state numbers to change semantics between runs | Applied **monotonic state sorting by mean GARCH volatility**: $\text{state\_order} = \text{argsort}([\bar{\sigma}_{GARCH}^{(0)}, \bar{\sigma}_{GARCH}^{(1)}, \bar{\sigma}_{GARCH}^{(2)}])$, guaranteeing 0=Low, 1=Mid, 2=High across every run |
| **Folds with no Mid/High days** (Folds 1 and 4) | Validation windows contain only Low days, so AUC is undefined; the code reports a 0.5 placeholder | Reported as undefined (Section 8). Ensembling does not change this. |
| **Docker cold-start model loading latency** | Loading XGBoost + HMM + scaler from disk on the first request was expected to add latency on Render's free tier (not measured) | Implemented **build-time model training** — models are trained and serialized during `docker build`, then loaded once per process on the first request with `@lru_cache(maxsize=1)`. Later requests reuse the cached objects (latency not benchmarked) |
| **Backward-looking HMM can't predict tomorrow** | The HMM labels each day using the full sample (Viterbi); it does not forecast | Added an XGBoost layer trained on $P(z_{T+1} = k \mid x_T)$. It scores 0.6515 pooled OOF AUC, below the 0.9455 of a "tomorrow = today" baseline, so treat it as a diagnostic, not a forecast. |
| **Unrealistic backtest assumptions in quant projects** | A backtest that omits fees, slippage and execution lag overstates returns | Enforced **realistic execution friction**: strict 1-day execution lag, 0.1% fees, 0.1% slippage per trade, with a constant $R_f = 6.5\%$ (a proxy for the India 10-year G-Sec yield) |

---

## 8. Results & Evaluation

### Predictive Model Validation (5-Fold Walk-Forward CV with 5-Day Gap)

Source: `results/metrics.json`. Each validation fold has 509 rows. Fold AUC is the macro average over the classes present in that fold.

| Fold | Fit / early-stop rows | Validation window | Val days Low / Mid / High | Macro-AUC |
|:---:|:---:|---|:---:|:---:|
| 1 | 423 / 76 | 2016-05-17 to 2018-06-06 | 509 / 0 / 0 | undefined (0.5 placeholder) |
| 2 | 856 / 152 | 2018-06-07 to 2020-07-09 | 474 / 8 / 27 | 0.9061 |
| 3 | 1,288 / 229 | 2020-07-10 to 2022-07-25 | 505 / 4 / 0 | 0.9460 |
| 4 | 1,721 / 305 | 2022-07-26 to 2024-08-21 | 509 / 0 / 0 | undefined (0.5 placeholder) |
| 5 | 2,154 / 381 | 2024-08-22 to 2026-09-10 | 497 / 12 / 0 | 0.8286 |
| **Pooled OOF** | | 2016-05-17 to 2026-09-10 (2,545 rows) | 2,494 / 24 / 27 | **0.6515** |

- Only 24 Mid and 27 High days exist in the validation windows, so every AUC rests on very few positive days. High appears only in fold 2.
- 0.6515 is the best of three sweep settings (`n_estimators` 200 / 300 / 500 gave 0.6515 / 0.6100 / 0.5958), so it is mildly optimistic.
- **Platform dependence**: the fold table and the 0.6515 come from a Windows run. The same code and data on Linux (the Docker build that Render serves) give pooled OOF AUC 0.699 and a different top-SHAP order (`vol_ratio_21_63` first on Linux, `gk_vol_21` first on Windows). The live dashboard therefore shows 0.699. The seed is fixed (`random_state=42 + fold`), the cause is not investigated, and both values are below the 0.9455 baseline.
- The mean of the five fold AUCs is 0.7362, but it includes two placeholders and is not used as the headline.
- **Baseline**: predicting tomorrow's regime = today's regime scores 0.9455 pooled macro-AUC on the same 2,545 rows (the regime changes on only 9 of them). The XGBoost layer does not beat it. Saved as `persistence_baseline_auc` in `results/metrics.json`.

### Backtest Performance Comparison (2014-04-07 to 2026-09-11, Rf = 6.5%)

Source: `results/results.json` (3,055 sessions, one pinned run).

| Metric | Regime strategy | Nifty 50 Buy & Hold | Difference |
|---|:---:|:---:|:---:|
| Sharpe (Rf 6.5%, 246 sessions/yr) | 0.269 | 0.305 | -0.036 |
| Max drawdown | -33.69% | -38.44% | +4.75 pts |
| Total return | +219.23% | +249.48% | -30.25 pts |
| CAGR | 9.79% | 10.59% | -0.80 pts |
| Calmar | 0.291 | 0.276 | +0.015 |
| Trades | 2 | 1 | |

- The strategy earns less than buy-and-hold and has a lower Sharpe. Its benefit is a max drawdown 4.75 points smaller.
- With only 27 High Vol days it trades twice in 12 years, so the result is close to buy-and-hold.
- Signals are shifted one day; the strategy pays 0.1% fees and 0.1% slippage per trade (buy-and-hold pays none); annualisation uses the sessions per year in the data (vectorbt would assume 365).

### Automated CI Test Suite (24/24 Tests Passing)

```bash
pytest tests/ -v
```

API and model tests read `data/`, `models/` and `results/`, so run `python run.py` and `python scripts/mlflow_sweep.py` first.

- Features (5): Garman-Klass output shape, Parkinson non-negative, 15 feature columns present, no nulls, rows dropped after warm-up.
- GARCH (3): `garch_vol` column exists, has values, strictly positive.
- HMM (3): 3 distinct regimes, labels in {0, 1, 2}, no nulls.
- Backtest (2): summary values are floats, final portfolio value is positive.
- Models (4): bundle keys and 5 fold models, AUC inside [0.5, 1], 15 feature names, `metrics.json` keys and 5 folds.
- API (7): `/` and 6 data endpoints return 200 with the expected keys.
- Not tested: data download and cache TTL, GARCH stationarity, HMM state ordering, the walk-forward gap, backtest lag.

---

## 9. Project Structure

```text
volatility-intelligence-platform/
├── app/
│   ├── main.py                  # FastAPI REST server (7 endpoints)
│   └── dashboard.py             # Streamlit exploration dashboard
├── data/
│   ├── nifty_ohlcv.csv          # OHLCV snapshot (7-day cache TTL)
│   ├── labeled_data.csv         # 15 features + GARCH vol + HMM regimes
│   └── backtest_summary.csv     # Strategy vs Buy & Hold statistics
├── frontend/
│   ├── pages/
│   │   ├── _app.tsx             # Next.js app wrapper
│   │   └── index.tsx            # Dashboard page; every number comes from the API
│   ├── package.json
│   └── tailwind.config.js
├── research/
│   ├── 01_arima_null_result.ipynb   # ADF, ARIMA(1,0,1) and ARCH tests on Nifty returns
│   └── README.md
├── results/
│   ├── results.json             # backtest, regime counts, environment (run.py)
│   └── metrics.json             # CV metrics, persistence baseline, top SHAP (scripts/)
├── scripts/
│   ├── regime_classifier.py     # XGBoost walk-forward CV and training
│   ├── mlflow_tracking.py       # MLflow experiment logger
│   └── mlflow_sweep.py          # 3-run sweep (n_estimators 200 / 300 / 500)
├── tests/                       # 24 pytest tests
│   ├── test_models.py
│   ├── test_features.py
│   ├── test_garch.py
│   ├── test_hmm.py
│   ├── test_backtest.py
│   └── test_api.py
├── .github/workflows/ci.yml     # run.py -> classifier -> pytest
├── run.py                       # pipeline entry point (writes results/results.json)
├── data.py                      # data ingestion and caching
├── features.py                  # 15 quantitative features
├── garch_model.py               # GARCH(1,1) conditional volatility
├── hmm_model.py                 # 3-state Gaussian HMM
├── backtest.py                  # vectorbt backtest
├── Dockerfile                   # single stage; trains models at build time
├── docker-compose.yml           # local run
├── render.yaml                  # Render blueprint
├── requirements.txt             # pinned with ==
├── pytest.ini
└── README.md

Generated and not committed (see .gitignore): models/xgb_bundle.pkl, outputs/*.png, mlflow.db, mlruns/.
```

---

## 10. Getting Started

### 1. Environment Setup

```bash
git clone https://github.com/RaajitSingh1306/volatility-intelligence-platform.git
cd "volatility-intelligence-platform"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Training Pipeline

```bash
python run.py                    # pinned data 2014-01-01..2026-09-12 -> features -> GARCH -> HMM -> backtest; writes results/results.json
python scripts/mlflow_sweep.py   # 3 XGBoost runs (n_estimators 200/300/500); saves best bundle and results/metrics.json
```

Reported run: 3,055 rows, 2014-04-07 to 2026-09-11. Python 3.12.10, yfinance 1.2.0, pandas 3.0.5, numpy 2.5.3, arch 8.0.0, hmmlearn 0.3.3, scikit-learn 1.9.0, shap 0.52.0, vectorbt 1.1.0, xgboost 3.3.0 (also saved in `results/results.json`). Every number in Section 8 comes from this single run.

### 3. Run Automated Tests

```bash
pytest tests/ -v
```

### 4. Launch FastAPI Backend

```bash
uvicorn app.main:app --reload --port 8000
```

- API Base URL: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`

### 5. Launch Next.js Frontend

```bash
cd frontend
npm install
npm run dev
```

- Web Console: `http://localhost:3000`

---

## 11. API Reference

### Endpoints

| Method | Endpoint | Description | Sample Output |
|---|---|---|---|
| `GET` | `/` | Health check | `{"status": "ok", "message": "Volatility Intelligence Platform API v2.0"}` |
| `GET` | `/current` | Latest regime, GARCH vol, close | fields: `date`, `regime`, `regime_name`, `garch_vol_pct`, `close` |
| `GET` | `/predict` | XGBoost ensemble probabilities for the next day | fields: `predicted_regime`, `predicted_regime_name`, `prob_low`, `prob_mid`, `prob_high`, `model_auc` |
| `GET` | `/market-summary` | Hero KPIs | fields: `current_regime`, `garch_vol_current`, `nifty_close`, `backtest_sharpe`, `backtest_max_dd`, `backtest_sharpe_bh`, `backtest_max_dd_bh`, `model_auc`, `persistence_baseline_auc`, `data_end_date`, `top_shap` (top 7 mean abs SHAP, from `results/metrics.json`; null or empty if `results/` is missing) |
| `GET` | `/regimes?last_n=252` | Regime history | list of `date`, `close`, `regime`, `regime_name`, `garch_vol_pct`, `returns_pct` |
| `GET` | `/stats` | Per-regime statistics | list of `regime_name`, `days`, `mean_vol_pct`, `mean_ret_pct`, `ret_std_pct` |
| `GET` | `/backtest-summary` | Strategy vs Buy & Hold table | `{metric: {"Strategy": x, "Buy & Hold": y}}`, e.g. `"Sharpe Ratio"`, `"Max Drawdown (%)"` |

---

## 12. Deployment

### Render Backend Deployment

The platform is deployed as a Docker Web Service on Render using `render.yaml`:
- Build: `docker build` runs `run.py` and `scripts/regime_classifier.py --n_estimators 200`, so trained models are baked into the image. Not verified by me; CI runs the same two commands.
- Startup: `uvicorn app.main:app`; data and models load on the first request, then stay cached in memory with `@lru_cache`.
- **Deploys are manual**: a push to GitHub does not redeploy Render for this service. After each push, open the Render dashboard and run Manual Deploy > Deploy latest commit (the Docker build trains the models, so it takes a few minutes). Vercel redeploys on every push, so deploy Render too, or the frontend will talk to an older API.

### Vercel Frontend Deployment

The Next.js frontend is deployed on Vercel:
- Configured with `NEXT_PUBLIC_API_URL=https://volatility-intelligence-platform.onrender.com`.
- Includes automatic cold-start retry handling for backend spinning instances.

---

## 13. Connected Portfolio Projects

- **[SEBI RAG Bot](https://github.com/RaajitSingh1306/sebi-rag-bot)**: Separate project. Its README says its `quant_agent` calls this API's `/predict` and `/current` (not verified here).
- **[NSEI Daily Stock Pipeline](https://github.com/RaajitSingh1306/NSEI-Daily-Stock-Pipeline)**: Separate project. VIP downloads its own data with `yfinance` (`data.py`) and does not read this pipeline.
- **[Nifty Sector Rotation](https://github.com/RaajitSingh1306/Nifty-Sector-Rotation)**: Separate cross-sectional momentum project. VIP does not call it and it is not wired to VIP's regime signals.
- **[Nifty-Time-Series](https://github.com/RaajitSingh1306/Nifty-Time-Series)**: Econometric analysis demonstrating the random-walk nature of returns that motivated VIP's variance modeling.

---

## 14. Limitations & Known Issues

- **Single Asset Focus**: Evaluates the Nifty 50 index (`^NSEI`); does not currently run multi-asset cross-market regimes (such as Bank Nifty or mid-caps).
- **Daily EOD Granularity**: Models daily closing bars; does not capture intraday regime shifts during morning gap-down market openings.
- **Lopsided regimes**: 97.3% Low (2,971 days), 1.9% Mid (57), 0.9% High (27 days, all Mar-Apr 2020). The HMM mainly isolates the COVID crash.
- **Look-ahead in labels**: the HMM is fit and decoded (Viterbi) on the full sample, so the label on day t uses later data. Backtest signals and XGBoost targets inherit this.
- **Predictive layer does not beat persistence**: pooled OOF Macro-AUC 0.6515 against 0.9455 for "tomorrow = today".
- **Undefined folds**: folds 1 and 4 have only Low days in validation, so their AUC cannot be computed.
- **Results differ by platform**: the same seeded code gives AUC 0.6515 on Windows and 0.699 on Linux (Section 8). Reported numbers name the platform they came from.
- **Labels are not volatility bands**: GARCH vol overlaps across labels (Low 8.7-47.9%, Mid 9.2-45.3%, High 47.4-92.0%), so a Mid Vol day can have the same GARCH vol as a Low day. The HMM clusters all 15 features, not volatility alone.
- **Backtest**: 2 trades in 12 years; lower return and Sharpe than buy-and-hold, 4.75 points less max drawdown.
- **Free-Tier Cold Starts**: Render free-tier instances sleep after 15 minutes of inactivity; the frontend includes polling retry logic to handle 50-second wakeups gracefully.

---

## 15. Roadmap / Future Expansion

- [ ] **Multi-Asset Cross-Market Regimes**: Expand HMM and XGBoost models to Bank Nifty, Nifty IT, and US S&P 500.
- [ ] **Intraday 15-Minute Regime Tracking**: Integrate streaming tick data for real-time intraday regime change alerts.
- [ ] **Reinforcement Learning Dynamic Sizing**: Replace the all-in / all-out rule (hold on Low and Mid, cash on High) with PPO-based position sizing optimized for Sortino ratio.
- [ ] **Automated Order Execution**: Integrate Zerodha Kite Connect API for live paper trading and execution.

---

## 16. License & Disclaimer

### License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

### Disclaimer
This software is intended strictly for quantitative finance research, algorithmic modeling education, and portfolio analytics. It does not constitute investment advice or trading recommendations.
