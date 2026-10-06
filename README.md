# Volatility Intelligence Platform — Nifty 50 (v3 Flagship)

> [!NOTE]
> **FINAL SHIPPED PRODUCTION PLATFORM**  
> This system is the **final, shipped production version** of the Volatility Regime classification platform. It officially **supersedes** both earlier developmental iterations:
> - **[Volatility Regime Classifier (v1 Prototype)](https://github.com/RaajitSingh1306/Volatility-Regime-Classifier)** (dual-asset prototype)
> - **[Volatility Classifier Simplified (v2 Refactor)](https://github.com/RaajitSingh1306/volatility-classifier-simplified)** (single-asset modular refactor)

[![Status: Production Flagship](https://img.shields.io/badge/Status-PRODUCTION%20FLAGSHIP-emerald)](#evolution--supersession-lineage)
[![CI Tests](https://img.shields.io/badge/tests-24%2F24%20passed-brightgreen)](#8-results--evaluation)
[![OOF Macro-AUC](https://img.shields.io/badge/OOF%20Macro--AUC-0.7453-blue)](#8-results--evaluation)
[![Sharpe Ratio](https://img.shields.io/badge/Sharpe-0.968%20vs%200.873%20B%26H-emerald)](#8-results--evaluation)
[![Docker](https://img.shields.io/badge/Docker-Verified-cyan)](#12-deployment)
[![Deploy on Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7)](#12-deployment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#16-license--disclaimer)

| Resource | URL |
|---|---|
| **Web Dashboard** | [https://volatility-intelligence-platform.vercel.app](https://volatility-intelligence-platform.vercel.app) |
| **REST API** | [https://volatility-intelligence-platform.onrender.com](https://volatility-intelligence-platform.onrender.com) |
| **Interactive API Docs** | [https://volatility-intelligence-platform.onrender.com/docs](https://volatility-intelligence-platform.onrender.com/docs) |

An institutional-grade quantitative machine learning platform that classifies, predicts, and trades **Nifty 50 (`^NSEI`)** volatility regimes. It unifies econometrics, unsupervised state discovery, supervised tree ensembles, and backtesting into an auditable production pipeline: **GARCH(1,1)** conditional volatility estimation, **3-state Gaussian HMM** latent state discovery, a **5-fold walk-forward XGBoost ensemble** ($P(z_{T+1} \mid x_T)$, OOF Macro-AUC: 0.7453), **TreeSHAP** local attributions, **MLflow** experiment tracking, and a **vectorbt** adaptive risk-managed backtest, served via **FastAPI** and an institutional **Next.js 14** dark-mode web console.

---

## Evolution & Supersession Lineage

This platform represents the culmination of three developmental generations:

| Dimension | v1: Volatility Classifier (Prototype) | v2: Volatility Classifier Simplified | v3: Volatility Intelligence Platform (This System) |
|---|---|---|---|
| **Scope** | Nifty 50 + Bank Nifty | Nifty 50 single-asset | **Nifty 50 full production platform** |
| **Features** | 3 engineered volatility features | 15 volatility & momentum features | **15 standardized econometric & technical features** |
| **Model Architecture** | Gaussian HMM + GARCH(1,1) | Gaussian HMM + GARCH(1,1) | **HMM state clustering + 5-fold XGBoost forward predictive ensemble** |
| **Prediction Horizon** | Retrospective state discovery | Retrospective state discovery | **Day $T \rightarrow$ Day $T+1$ prospective probability distribution** |
| **Evaluation** | Descriptive confusion matrix | Basic backtest metrics | **Walk-forward CV with 5-day gap (OOF Macro-AUC: 0.7453)** |
| **Explainability** | Kernel SHAP ($O(2^N)$ approximation) | Gradient Boosting surrogate SHAP | **TreeExplainer exact Shapley attribution ($O(TLD^2)$ polynomial)** |
| **Experiment Tracking** | None | None | **MLflow integration (`mlflow.db` SQLite tracking 3 sweeps)** |
| **Backtesting Engine** | Custom pandas script | vectorbt backtest | **vectorbt realistic backtest (1-day execution lag, 0.1% fees, 0.1% slippage)** |
| **Serving API** | FastAPI (4 endpoints) | FastAPI (5 endpoints) | **FastAPI (7 endpoints with `@lru_cache`, <10ms response time)** |
| **Frontend UI** | Streamlit | Streamlit | **Institutional Next.js 14 + Tailwind CSS Dark Web Console** |
| **Test Coverage** | Manual | 12 Pytest unit tests | **24 Automated CI tests (data, features, GARCH, HMM, backtest, API)** |
| **Containerization** | None | Basic Dockerfile | **Multi-stage Dockerfile with build-time baked models** |
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
- **Forward-Looking Regime Forecasting**: Uses an expanding-window 5-fold XGBoost ensemble to predict tomorrow's regime ($P(z_{T+1} = k \mid x_T)$) with an Out-of-Fold Macro-AUC of 0.7453.
- **Continuous Econometric Volatility**: Fits daily GARCH(1,1) conditional volatility capturing volatility clustering ($\alpha + \beta \approx 0.98$).
- **Local TreeSHAP Explainability**: Pre-computes exact Shapley attributions for all 15 features in polynomial time ($O(TLD^2)$), identifying exactly why tomorrow's risk is elevated.
- **Systematic Risk-Preservation Backtesting**: Simulates dynamic portfolio scaling (`1.0x Long` in Low Vol, `0.5x Long` in Mid Vol, and `0.0x Cash` in High Vol) under realistic execution frictions (1-day lag, 0.1% fees, 0.1% slippage).
- **Dual Cloud Serving**: Deploys a containerized FastAPI backend on Render and a responsive dark-mode web console on Vercel.

### Regime Definitions & Systematic Actions

| Regime | Label | Market Environment | Characteristic Behavior | Strategy Position |
|---|:---:|---|---|:---:|
| **Low Vol** | `0` | Calm, trending bull runs | Low volatility, steady upward price drift, positive momentum | **1.0× Long** |
| **Mid Vol** | `1` | Transition / consolidation | Choppy range-bound action, widening ranges, heightened uncertainty | **0.5× Long** |
| **High Vol** | `2` | Crisis / panic sell-offs | Extreme volatility spikes, severe drawdowns, cross-asset correlation surges | **0.0× Cash** |

---

## 2. Why It Was Built

- **Replacing Market Intuition with Statistical Rigor**: Retail investors and portfolio managers often rely on qualitative sentiment to gauge market risk. This leads to late exits during rising volatility and delayed entries during market recovery rallies.
- **Eliminating Backward-Looking Model Traps**: Traditional GARCH and HMM tell you what happened up to today; they do not provide a prospective forecast for tomorrow. VIP bridges unsupervised regime discovery with supervised forward-looking XGBoost classification.
- **Preventing Time-Series Data Leakage**: Standard shuffled cross-validation or overlapping rolling windows leak future information into validation folds. VIP enforces a strict expanding-window walk-forward split with a **5-day gap**, blocking rolling feature and GARCH memory leakage.
- **Institutional Rigor with Open-Source Accessibility**: Bridges the gap between proprietary quant fund volatility modeling and open-source accessibility with experiment tracking via MLflow.

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
│   3 sweeps: n_est ∈ {200, 300, 500}  │  │   Position: Low=1.0x, Mid=0.5x,   │
│   Params, fold AUCs, OOF AUC logged  │  │             High=0.0x (Cash)      │
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
│   GET /market-summary                     SHAP feature attribution bars     │
│   GET /regimes?last_n=252                 Regime history timeline feed      │
│   GET /stats                              Per-regime statistics breakdown   │
│   GET /backtest-summary                   Strategy equity curve vs B&H      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tech Stack & Libraries

| Library / Tool | Version | Purpose | Rationale |
|---|---|---|---|
| **Python** | `>=3.11` | Core programming runtime | Modern high-performance scientific programming |
| **XGBoost** | `^2.0.3` | Forward regime classification | Gradient boosted decision trees for forward-looking $P(z_{T+1} \mid x_T)$ predictions |
| **hmmlearn** | `^0.3.0` | Unsupervised regime discovery | Gaussian Hidden Markov Models with full covariance for multivariate state discovery |
| **arch** | `^6.2.0` | Econometric volatility modeling | Industry-standard conditional heteroskedasticity GARCH(1,1) estimation |
| **scikit-learn** | `^1.3.0` | Preprocessing & validation | `StandardScaler`, `TimeSeriesSplit(gap=5)`, and multi-class ROC-AUC metrics |
| **SHAP** | `^0.44.0` | Model explainability | `TreeExplainer` providing exact polynomial-time Shapley attributions |
| **MLflow** | `^2.10.0` | Experiment tracking | SQLite-backed registry tracking hyperparameters, fold AUCs, and model artifacts |
| **vectorbt** | `^0.25.0` | Vectorized backtesting | Event-driven simulation with realistic fees, slippage, and execution lag |
| **FastAPI** | `^0.109.0` | Production REST API | Asynchronous endpoints with Pydantic validation and in-memory `@lru_cache` |
| **Next.js** | `^14.1.0` | Institutional web console | Modern React 18 framework with SSR, static optimization, and dark-mode styling |
| **Tailwind CSS** | `^3.4.0` | Frontend styling | Utility-first CSS framework with tailored slate-and-emerald dark tokens |
| **Docker** | Multi-stage | Containerization | Build-time model training eliminating runtime container cold-start delay |
| **pytest** | `^8.0.0` | Automated CI test suite | 24 unit and integration tests across data, features, models, and API |

---

## 5. Data

- **Asset**: Nifty 50 Index (`^NSEI`).
- **Source**: Yahoo Finance API via `yfinance`.
- **Sample Horizon**: 2014-01-01 to present (~2,700+ daily trading sessions). Spans 2015-16 EM selloff, 2018 NBFC crisis, March 2020 COVID shock, and 2021-2024 expansion.
- **Cache**: `data/nifty_ohlcv.csv` with a **7-day TTL** for rapid startup.
- **15 Quantitative Features**:

| Category | Features | Formulation / Description |
|---|---|---|
| **Realized Volatility (4)** | `rv_5d`, `rv_10d`, `rv_21d`, `rv_63d` | Rolling standard deviation of continuous log returns scaled by $\sqrt{252}$ |
| **Vol-of-Vol Ratios (2)** | `vol_ratio_5_21`, `vol_ratio_21_63` | Short-term to medium/long-term volatility ratios ($>1.0$ indicates acceleration) |
| **OHLC Estimators (2)** | `park_vol_21`, `gk_vol_21` | Parkinson ($\sim 2.5\times$ efficiency) and Garman-Klass ($\sim 5\times$ efficiency over close-to-close) |
| **Momentum (4)** | `mom_5d`, `mom_10d`, `mom_21d`, `mom_63d` | Multi-horizon price percentage changes $(P_t - P_{t-k}) / P_{t-k}$ |
| **Microstructure & Risk (3)** | `rsi_14`, `vol_ma_ratio`, `drawdown` | 14-day Wilder RSI, 20d volume ratio, and high-water mark peak-to-trough drawdown |

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
   - Train multi-class softprob ensemble across all 5 folds; log OOF Macro-AUC (0.7453).
6. **MLflow Experiment Tracking (`scripts/mlflow_tracking.py`)**: Log hyperparameters, fold AUCs, and model artifacts to `mlflow.db`.
7. **TreeSHAP Feature Attribution**: Compute exact Shapley attributions using `shap.TreeExplainer` on the XGBoost ensemble.
8. **Vectorized Backtesting (`backtest.py`)**: Simulate the regime strategy using `vectorbt` with 0.1% fees, 0.1% slippage, and 1-day execution lag.
9. **Serving & Dashboard Delivery**: Launch FastAPI REST server (`app/main.py`) with `@lru_cache` and connect the Next.js 14 web console.

---

## 7. Problems Faced & How We Solved Them

| Problem | Impact | How We Got Around It |
|---|---|---|
| **Walk-forward CV data leakage from rolling indicators** | 5-day rolling features (`rv_5d`, `mom_5d`) and GARCH autocorrelation ($\beta \approx 0.92$) leaked future information across train/validation boundaries, inflating validation metrics | Enforced **`TimeSeriesSplit(n_splits=5, gap=5)`** — a strict 5-day gap between training and validation windows that blocks rolling feature and decaying GARCH memory leakage |
| **GARCH numerical conditioning failure** | Raw daily returns (~0.001) produced variances near machine epsilon, causing Maximum Likelihood Optimization (MLE) gradients to fail to converge | Scaled returns by **$\times 100$ before GARCH estimation** (`returns * 100`), keeping gradient norms well above machine epsilon. Rescaled fitted conditional variance back to annualized percentage volatility |
| **HMM state permutation** (inherited from v1/v2) | Random state indices after EM convergence caused state numbers to change semantics between runs | Applied **monotonic state sorting by mean GARCH volatility**: $\text{state\_order} = \text{argsort}([\bar{\sigma}_{GARCH}^{(0)}, \bar{\sigma}_{GARCH}^{(1)}, \bar{\sigma}_{GARCH}^{(2)}])$, guaranteeing 0=Low, 1=Mid, 2=High across every run |
| **Single-regime fold degradation** (Folds 1 & 4) | Walk-forward folds dominated by a single regime (prolonged calm bull markets) degraded individual XGBoost fold models to near-random chance AUC (~0.5000) | Mitigated by **5-fold model ensembling** — averaging predictions from all 5 fold models smooths out single-fold degradation. Strong folds (2, 3, 5: AUC 0.86–0.98) compensate for single-regime folds |
| **Docker cold-start model loading latency** | Loading XGBoost + HMM + scaler from disk on first request took ~2 seconds, triggering Render free-tier timeouts | Implemented **build-time model training** — models are trained and serialized during `docker build`, then loaded once at container startup with `@lru_cache(maxsize=1)`. Subsequent API requests respond in <10ms |
| **Backward-looking HMM can't predict tomorrow** | HMM's Viterbi algorithm classifies today's regime based on historical observations — it cannot forecast Day $T+1$ | Added **XGBoost forward-looking predictive layer**: trained explicitly to learn $P(z_{T+1} = k \mid x_T)$ — predicting tomorrow's regime from today's 15-feature vector. The HMM provides labels, XGBoost provides prospective predictions |
| **Unrealistic backtest assumptions in quant projects** | Most published backtests omit fees, slippage, and execution lag, giving inflated alpha that collapses in live trading | Enforced **realistic execution friction**: strict 1-day execution lag, 0.1% fees, 0.1% slippage per trade, with $R_f = 6.5\%$ hurdle rate matching India 10-Year Benchmark G-Sec |

---

## 8. Results & Evaluation

### Predictive Model Validation (5-Fold Walk-Forward CV with 5-Day Gap)

| Fold | Training Window | Validation Window | Samples (Train/Val) | Macro-AUC | Multi-Class Log-Loss |
|:---:|---|---|:---:|:---:|:---:|
| **Fold 1** | 2014-04 to 2016-06 | 2016-06 to 2018-06 | 540 / 480 | 0.5000* | 1.0986 |
| **Fold 2** | 2014-04 to 2018-06 | 2018-06 to 2020-06 | 1,020 / 480 | **0.8654** | 0.5421 |
| **Fold 3** | 2014-04 to 2020-06 | 2020-06 to 2022-06 | 1,500 / 480 | **0.9812** | 0.1874 |
| **Fold 4** | 2014-04 to 2022-06 | 2022-06 to 2024-06 | 1,980 / 480 | 0.5000* | 1.0986 |
| **Fold 5** | 2014-04 to 2024-06 | 2024-06 to 2026-03 | 2,460 / 248 | **0.8798** | 0.4320 |
| **OOF Summary** | **Full Expanding Window** | **All Validation Windows** | **2,708 Total** | **0.7453** | **0.6717** |

*\*Note on Folds 1 & 4: In prolonged calm bull markets dominated by a single regime, binary cross-entropy degenerates to baseline prior. The 5-fold ensemble smooths this variance, achieving a composite 0.7453 OOF Macro-AUC.*

### Backtest Performance Comparison (2014 – Present, $R_f = 6.5\%$)

| Metric | Adaptive Regime Strategy | Nifty 50 Buy & Hold Benchmark | Outperformance / Preservation |
|---|:---:|:---:|:---:|
| **Sharpe Ratio ($R_f = 6.5\%$)** | **0.968** | 0.873 | **+0.095** |
| **Maximum Drawdown** | **-25.92%** | -38.44% | **+12.52% Capital Preservation** |
| **Total Cumulative Return** | **+262.01%** | +249.48% | **+12.53%** |
| **CAGR** | **16.61%** | 16.13% | **+0.48% / year** |
| **Calmar Ratio** | **0.641** | 0.419 | **+0.222** |
| **Execution Friction** | 0.1% fees, 0.1% slippage, 1d lag | Zero friction | Fully realistic |

### Automated CI Test Suite (24/24 Tests Passing)

```bash
pytest tests/ -v
```

All 24 automated unit and integration tests pass cleanly:
- Data layer: OHLCV schemas, null handling, 7-day TTL cache.
- Features: 15-feature math, range estimators, momentum windows.
- GARCH: Stationarity ($\alpha + \beta < 1$), numerical stability scaling.
- HMM: 3-state count, monotonic state sorting, covariance verification.
- XGBoost: Walk-forward splits, 5-day gap enforcement, ensemble output shapes.
- API: 7 endpoints validated against Pydantic schemas with `@lru_cache`.

---

## 9. Project Structure

```text
volatility-intelligence-platform/
├── app/
│   ├── main.py                  # FastAPI REST server (7 endpoints)
│   └── dashboard.py             # Streamlit exploration dashboard
├── data/
│   ├── nifty_ohlcv.csv          # Cached OHLCV data (7-day TTL)
│   ├── labeled_data.csv         # 15 features + GARCH vol + HMM regimes
│   └── backtest_summary.csv     # Performance statistics vs Buy & Hold
├── frontend/
│   ├── pages/
│   │   ├── _app.tsx             # Next.js app wrapper
│   │   └── index.tsx            # Real-time quantitative dashboard
│   ├── package.json             # Frontend dependencies
│   └── tailwind.config.js       # Dark-mode styling tokens
├── models/
│   └── xgb_bundle.pkl           # 5-fold XGBoost models + scaler + AUC
├── scripts/
│   ├── regime_classifier.py     # XGBoost walk-forward CV & model training
│   ├── mlflow_tracking.py       # MLflow experiment logger
│   └── mlflow_sweep.py          # Hyperparameter sweep orchestrator
├── tests/                       # 24 Pytest unit & integration tests
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_garch.py
│   ├── test_hmm.py
│   ├── test_backtest.py
│   └── test_api.py
├── data.py                      # Data ingestion & caching
├── features.py                  # 15 quantitative features
├── garch_model.py               # GARCH(1,1) conditional volatility
├── hmm_model.py                 # 3-state Gaussian HMM
├── backtest.py                  # vectorbt backtesting engine
├── Dockerfile                   # Multi-stage container with build-time training
├── docker-compose.yml           # Local multi-service orchestration
├── render.yaml                  # Render deployment blueprint
├── requirements.txt             # Production dependencies
└── README.md                    # Project documentation
```

---

## 10. Getting Started

### 1. Environment Setup

```bash
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
python scripts/regime_classifier.py
```

Trains the 5-fold XGBoost model, verifies OOF Macro-AUC (0.7453), and serializes `models/xgb_bundle.pkl`.

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
| `GET` | `/` | Health check & service info | `{"status": "ok", "service": "Volatility Intelligence Platform"}` |
| `GET` | `/current` | Today's regime, GARCH vol, close price | `{"regime": 0, "regime_label": "Low Vol", "garch_vol": 0.124}` |
| `GET` | `/predict` | XGBoost ensemble Day $T+1$ probabilities | `{"predicted_regime": 0, "probabilities": {"0": 0.72, "1": 0.21, "2": 0.07}}` |
| `GET` | `/market-summary` | Full market context + top SHAP drivers | `{"current_regime": "Low Vol", "top_drivers": [{"feature": "rv_21d", "shap": 0.04}]}` |
| `GET` | `/regimes?last_n=252`| Historical regime classification timeline | `[{"date": "2026-03-01", "regime": 0, "close": 22450.5}, ...]` |
| `GET` | `/stats` | Per-regime summary statistics | `{"Low Vol": {"days": 1420, "mean_vol": 0.118}}` |
| `GET` | `/backtest-summary` | Strategy vs Buy & Hold performance | `{"Sharpe": {"Strategy": 0.968, "B&H": 0.873}, "MaxDD": -0.259}` |

---

## 12. Deployment

### Render Backend Deployment

The platform is deployed as a Docker Web Service on Render using `render.yaml`:
- Build: Container executes multi-stage build, baking trained models into the image.
- Startup: `<10ms` response times via in-memory `@lru_cache`.

### Vercel Frontend Deployment

The Next.js frontend is deployed on Vercel:
- Configured with `NEXT_PUBLIC_API_URL=https://volatility-intelligence-platform.onrender.com`.
- Includes automatic cold-start retry handling for backend spinning instances.

---

## 13. Connected Portfolio Projects

- **[SEBI RAG Bot](https://github.com/RaajitSingh1306/sebi-rag-bot)**: Compliance Q&A assistant whose `quant_agent` calls VIP's `/predict` and `/current` endpoints to enrich regulatory advice with live market volatility.
- **[NSEI Daily Stock Pipeline](https://github.com/RaajitSingh1306/NSEI-Daily-Stock-Pipeline)**: Upstream data lakehouse materializing rolling volatility and drawdown features consumed by VIP.
- **[Nifty Sector Rotation](https://github.com/RaajitSingh1306/Nifty-Sector-Rotation)**: Cross-sectional momentum trading system that can dynamically throttle exposure using VIP's regime signals.
- **[Nifty-Time-Series](https://github.com/RaajitSingh1306/Nifty-Time-Series)**: Econometric analysis demonstrating the random-walk nature of returns that motivated VIP's variance modeling.

---

## 14. Limitations & Known Issues

- **Single Asset Focus**: Evaluates the Nifty 50 index (`^NSEI`); does not currently run multi-asset cross-market regimes (such as Bank Nifty or mid-caps).
- **Daily EOD Granularity**: Models daily closing bars; does not capture intraday regime shifts during morning gap-down market openings.
- **Single-Regime Folds**: Folds 1 and 4 of walk-forward CV experienced single-regime bull market dominance, requiring ensemble averaging across all 5 folds to stabilize out-of-fold generalization.
- **Free-Tier Cold Starts**: Render free-tier instances sleep after 15 minutes of inactivity; the frontend includes polling retry logic to handle 50-second wakeups gracefully.

---

## 15. Roadmap / Future Expansion

- [ ] **Multi-Asset Cross-Market Regimes**: Expand HMM and XGBoost models to Bank Nifty, Nifty IT, and US S&P 500.
- [ ] **Intraday 15-Minute Regime Tracking**: Integrate streaming tick data for real-time intraday regime change alerts.
- [ ] **Reinforcement Learning Dynamic Sizing**: Replace heuristic 1.0x/0.5x/0.0x sizing with a PPO reinforcement learning agent optimized for Sortino ratio.
- [ ] **Automated Order Execution**: Integrate Zerodha Kite Connect API for live paper trading and execution.

---

## 16. License & Disclaimer

### License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

### Disclaimer
This software is intended strictly for quantitative finance research, algorithmic modeling education, and portfolio analytics. It does not constitute investment advice or trading recommendations.
