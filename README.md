# Volatility Intelligence Platform — Nifty 50

> [!NOTE]
> **FINAL SHIPPED PRODUCTION VERSION**  
> This system is the **final, shipped production version** of the Volatility Regime classification platform. It officially **supersedes** both earlier developmental iterations:
> - **[Volatility Regime Classifier (v1 Prototype)](https://github.com/RaajitSingh1306/Volatility-Regime-Classifier)** (dual-asset prototype)
> - **[Volatility Classifier Simplified (v2 Refactor)](https://github.com/RaajitSingh1306/volatility-classifier-simplified)** (single-asset modular refactor)

[![CI Tests](https://img.shields.io/badge/tests-24%2F24%20passed-brightgreen)](#testing)
[![OOF Macro-AUC](https://img.shields.io/badge/OOF%20Macro--AUC-0.7453-blue)](#predictive-model-xgboost-walk-forward-cv)
[![Sharpe Ratio](https://img.shields.io/badge/Sharpe-0.968%20vs%200.873%20B%26H-emerald)](#backtest-performance-2014--present-nifty-50)
[![Docker](https://img.shields.io/badge/Docker-Verified-cyan)](#docker-recommended)
[![Deploy on Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7)](#deployment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

| | |
|---|---|
| **Web Dashboard** | https://volatility-intelligence-platform.vercel.app |
| **REST API** | https://volatility-intelligence-platform.onrender.com |
| **API Docs** | https://volatility-intelligence-platform.onrender.com/docs |

---

## Table of Contents

1. [What](#what)
   - [Overview](#overview)
   - [Regime Definitions](#regime-definitions)
   - [Platform Outputs](#platform-outputs)
2. [Why](#why)
   - [The Problem](#the-problem)
   - [The Solution](#the-solution)
   - [Architectural Decisions & Trade-offs](#architectural-decisions--trade-offs)
   - [Simplified vs. Intelligence Platform](#how-this-differs-from-the-simplified-version)
3. [How](#how)
   - [End-to-End System Architecture](#end-to-end-system-architecture)
   - [1. Data Layer](#1-data-layer)
   - [2. Feature Engineering & Mathematical Foundations](#2-feature-engineering--mathematical-foundations)
   - [3. Model Chain & Formulations](#3-model-chain--formulations)
     - [3.1 GARCH(1,1) Conditional Volatility](#31-garch11-conditional-volatility)
     - [3.2 3-State Gaussian Hidden Markov Model](#32-3-state-gaussian-hidden-markov-model)
     - [3.3 XGBoost Forward-Looking Predictive Layer](#33-xgboost-forward-looking-predictive-layer)
     - [3.4 TreeExplainer SHAP Feature Attribution](#34-treeexplainer-shap-feature-attribution)
   - [4. Experiment Tracking (MLflow)](#4-experiment-tracking-mlflow)
   - [5. Backtesting Framework](#5-backtesting-framework)
   - [6. Serving Layer](#6-serving-layer)
4. [Where](#where)
   - [Project Directory Layout](#project-directory-layout)
   - [REST API Reference](#rest-api-reference)
   - [Getting Started](#getting-started)
   - [Evaluation & Benchmarks](#evaluation--benchmarks)
   - [Deployment](#deployment)
   - [Connected Projects](#connected-projects)
   - [Limitations & Roadmap](#limitations--roadmap)
   - [License & Disclaimer](#license--disclaimer)

---

## What

### Overview

The Volatility Intelligence Platform is a **production-grade quantitative machine learning system** that classifies, predicts, and trades Nifty 50 volatility regimes. It unifies econometrics, unsupervised state discovery, supervised tree ensembles, and backtesting into an auditable pipeline:

1. **GARCH(1,1)** estimates daily conditional volatility
2. **3-state Gaussian HMM** uncovers latent market regimes (Low, Mid, High)
3. **XGBoost ensemble** predicts tomorrow's regime from today's market signals
4. **vectorbt backtest** simulates an adaptive capital-preservation strategy against Nifty 50 Buy & Hold

Predictions and indicators are served via a **FastAPI REST API** with 7 endpoints and visualized in a **Next.js web dashboard** with real-time KPI cards, regime probability distributions, SHAP attribution, and equity curves.

### Regime Definitions

| Regime | Label | Market Conditions | Characteristic Behavior | Strategy Position |
|---|---:|---|---|---|
| **Low Vol** | `0` | Calm, trending bull runs | Low volatility, steady upward price drift, positive momentum | **1.0× Long** |
| **Mid Vol** | `1` | Transition / consolidation | Choppy range-bound action, widening ranges, heightened uncertainty | **0.5× Long** |
| **High Vol** | `2` | Crisis / panic sell-offs | Extreme volatility spikes, severe drawdowns, high cross-asset correlation | **0.0× Cash** |

### Platform Outputs

| Output | Description | Delivering Component |
|---|---|---|
| **Current regime** | Today's volatility state (`Low`, `Mid`, `High`) based on HMM classification | `GET /current` |
| **Next-day prediction** | XGBoost 5-fold ensemble probability distribution over tomorrow's regime | `GET /predict` |
| **GARCH volatility** | Annualized conditional volatility estimate ($\sigma_t$) updated daily | `GET /current` |
| **SHAP explanations** | Exact Shapley values indicating which features drove tomorrow's prediction | `GET /market-summary` |
| **Backtest report** | Sharpe ratio, max drawdown, CAGR comparison vs Buy & Hold | `GET /backtest-summary` |
| **MLflow experiment log** | 3 hyperparameter sweeps with parameter lineage and metric tracking | `mlflow.db` / `mlruns/` |

---

## Why

### The Problem

Institutional quant funds deploy sophisticated volatility-modeling engines to dynamically scale risk before market drawdowns wipe out returns. Independent researchers and retail investors face severe barriers:

1. **Intuition instead of measurement** — "The market feels risky" is not an actionable, systematic trading signal.
2. **Backward-looking models** — Traditional GARCH and HMM tell you what happened up to today, but provide no prospective forecast for tomorrow.
3. **Pervasive data leakage** — Most open-source quant projects use standard shuffled cross-validation or overlapping rolling windows, reporting unrealistically high accuracy that collapses live.
4. **Unrealistic backtests** — Published backtests routinely omit fees, slippage, and execution lag, giving a false sense of alpha.
5. **No experiment lineage** — Ad-hoc hyperparameter tuning without structured experiment tracking makes results impossible to audit or reproduce.
6. **No production serving** — Models remain locked in Jupyter notebooks rather than running as containerized APIs.

### The Solution

| Problem | Solution | Implementation in Platform |
|---|---|---|
| No objective volatility measurement | **GARCH(1,1)** conditional volatility | Captures ARCH clustering effects ($\alpha+\beta \approx 0.98$) |
| Only backward-looking | **XGBoost forward predictive layer** | Learns mapping from Day $T$ features $\rightarrow$ Day $T+1$ regime |
| Data leakage in time series | **Strict walk-forward CV with 5-day gap** | `TimeSeriesSplit(gap=5)` blocks rolling feature & GARCH memory leakage |
| Artificially inflated backtests | **Realistic execution friction** | 1-day execution lag, 0.1% fees, 0.1% slippage |
| Undocumented hyperparameter tuning | **MLflow experiment tracking** | SQLite-backed `mlflow.db` logging parameters, fold AUCs, and artifacts |
| Fragile notebook code | **Dockerized FastAPI + Next.js** | 7 REST endpoints with `@lru_cache` and automated CI tests |

### Architectural Decisions & Trade-offs

| Decision | Implementation | Rationale |
|---|---|---|
| **HMM before XGBoost** | HMM labels $\rightarrow$ XGBoost targets | HMM discovers regimes unsupervised without arbitrary thresholds; XGBoost learns the temporal mapping to predict them forward. |
| **Gap in CV** | `TimeSeriesSplit(gap=5)` | Prevents GARCH memory and 5-day rolling window leakage into validation folds. |
| **Ensemble averaging** | Mean of 5 fold models | Smooths out-of-fold variance. Any single fold model may overfit to its specific regime history. |
| **1-day execution lag** | Signal at $T \rightarrow$ trade at $T+1$ | Eliminates same-day forward-looking execution bias. |
| **TreeExplainer** | `shap.TreeExplainer` | Exact Shapley values computed in $O(TLD^2)$ polynomial time instead of exponential KernelExplainer approximations. |
| **Full covariance HMM** | `covariance_type='full'` | Captures regime-specific feature correlations (e.g. vol-drawdown correlation strengthens during crises). |
| **State sorting** | Ascending sort by mean `garch_vol` | Removes HMM's inherent random state index permutation. Guarantees 0=Low, 1=Mid, 2=High across all runs. |
| **LRU caching** | `@lru_cache(maxsize=1)` | Single in-memory allocation for data and models. API requests respond in < 10 ms. |
| **Flat-file contracts** | CSV and PKL artifacts | Avoids heavyweight external database dependencies. Components can run and be tested independently in CI. |
| **Docker build-time training** | Baked models inside image | Container starts instantaneously on deployment with zero runtime model training dependencies. |
| **Return scaling** | `returns * 100` before GARCH | Well-conditioned maximum likelihood estimation. Raw returns produce variances near machine epsilon. |

### Evolution & Supersession Lineage

This platform is the culmination of three developmental generations:

| Dimension | v1: Volatility Classifier (Prototype) | v2: Volatility Classifier Simplified | v3: Volatility Intelligence Platform (Final Shipped Version) |
|---|---|---|---|
| **Scope** | Nifty 50 + Bank Nifty | Nifty 50 single-asset | Nifty 50 full production platform |
| **Features** | 3 engineered volatility features | 15 volatility & momentum features | 15 standardized econometrics & technical features |
| **Model Architecture** | Gaussian HMM + GARCH(1,1) | Gaussian HMM + GARCH(1,1) | **HMM unsupervised clustering + 5-fold XGBoost forward-looking ensemble** |
| **Prediction Horizon** | Current day classification only | Current day classification only | **Day $T \rightarrow Day T+1$ predictive probability distribution** |
| **Model Evaluation** | Descriptive confusion matrix | Basic backtest metrics | **Walk-forward CV with 5-day gap (OOF Macro-AUC: 0.7453)** |
| **Explainability** | Kernel SHAP ($O(2^N)$ approximation) | Gradient Boosting surrogate SHAP | **TreeExplainer exact Shapley attribution ($O(TLD^2)$ polynomial)** |
| **Experiment Tracking** | None | None | **MLflow integration (`mlflow.db` SQLite tracking 3 sweeps)** |
| **Backtesting Engine** | Custom pandas script | vectorbt backtest | **vectorbt realistic backtest (1-day execution lag, 0.1% fees, 0.1% slippage)** |
| **Serving API** | FastAPI (4 endpoints) | FastAPI (5 endpoints) | **FastAPI (7 endpoints with `@lru_cache`, <10ms response time)** |
| **Frontend UI** | Streamlit | Streamlit | **Institutional Next.js 14 + Tailwind CSS Dark UI** |
| **Test Coverage** | Manual | 12 Pytest unit tests | **24 Automated CI tests (data, features, GARCH, HMM, backtest, API)** |
| **Containerization** | None | Basic Dockerfile | **Multi-stage Dockerfile with bake-time model serialization** |
| **Status** | *Superseded* | *Superseded* | **Active Flagship Shipped Platform** |

---

## How

### End-to-End System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. Data Layer (yfinance)                           │
│   data.py → data/nifty_ohlcv.csv (7-day cache TTL)                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ OHLCV: Date, Open, High, Low, Close, Volume
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       2. Feature Engineering Layer                          │
│   features.py → 15 features across 5 categories                             │
│   Realized vol · Vol-of-vol · OHLC estimators · Momentum · Microstructure   │
└──────────────────┬─────────────────────────────────────┬────────────────────┘
                   │                                     │
                   ▼                                     ▼
┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
│          GARCH(1,1) Layer            │  │            HMM Layer              │
│   garch_model.py                     │  │   hmm_model.py                    │
│                                      │  │                                   │
│   σ²ₜ = ω + αε²ₜ₋₁ + βσ²ₜ₋₁          │  │   3-state Gaussian, full cov      │
│   → garch_vol series                 │  │   Sorted: 0=Low, 1=Mid, 2=High    │
└──────────────────┬───────────────────┘  └──────────────────┬────────────────┘
                   │                                         │
                   └───────────────────┬─────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       3. XGBoost Predictive Layer                           │
│   scripts/regime_classifier.py                                              │
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
│        4. MLflow Tracking            │  │        5. vectorbt Backtest       │
│   scripts/mlflow_tracking.py         │  │   backtest.py                     │
│   mlflow.db (SQLite)                 │  │                                   │
│   3 sweeps: n_est ∈ {200, 300, 500}  │  │   Position: Low=1.0x, Mid=0.5x,   │
│   Params, fold AUCs, OOF AUC         │  │             High=0.0x (Cash)      │
│   → mlruns/ · mlflow.db              │  │   Friction: 1-day lag, 0.1% fees  │
└──────────────────────────────────────┘  └──────────────────┬────────────────┘
                                                             │
                                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          6. Serving Layer                                   │
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

### 1. Data Layer

**File:** `data.py`

* **Asset**: Nifty 50 Index (`^NSEI`) via Yahoo Finance
* **Sample Period**: 2014-01-01 to present (~2,700+ trading days)
* **Columns**: Open, High, Low, Close, Volume
* **Returns**: Daily simple returns ($r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$) and log returns ($\ln \frac{P_t}{P_{t-1}}$)
* **Local Cache**: Saved to `data/nifty_ohlcv.csv` with a **7-day TTL** to optimize startup and CI speeds.

**Why 2014 to present?**
This window spans major macroeconomic events: the 2015-16 emerging market selloff, 2018 NBFC liquidity crisis, the March 2020 COVID crash, and the 2021-2024 expansion. This diversity ensures sufficient representations of rare High Vol crisis regimes.

---

### 2. Feature Engineering & Mathematical Foundations

**File:** `features.py`

The platform computes **15 quantitative features** across five distinct categories:

#### 2.1 Realized Volatility (4 features)
Rolling standard deviation of log returns annualized to 252 trading days:

$$RV_n = \text{std}(r_{t-n+1}, \dots, r_t) \times \sqrt{252}$$

* `rv_5d`: 5-day realized volatility (weekly)
* `rv_10d`: 10-day realized volatility (bi-weekly)
* `rv_21d`: 21-day realized volatility (monthly baseline)
* `rv_63d`: 63-day realized volatility (quarterly trend)

#### 2.2 Volatility-of-Volatility (2 features)
Ratios capturing acceleration or deceleration in variance:

$$\text{vol\_ratio\_5\_21} = \frac{RV_5}{RV_{21}}, \quad \text{vol\_ratio\_21\_63} = \frac{RV_{21}}{RV_{63}}$$

Values $> 1.0$ indicate rapid short-term volatility expansion, often signaling regime transition.

#### 2.3 OHLC Range-Based Estimators (2 features)
Close-to-close volatility ignores intraday dynamics. Range estimators provide higher statistical efficiency:

* **Garman-Klass Volatility (`gk_vol_21`)** — $\sim 5\times$ more efficient than close-to-close variance:

  $$\sigma_{GK}^2 = \frac{1}{n} \sum_{i=1}^n \left[ 0.5 \left(\ln \frac{H_i}{L_i}\right)^2 - (2\ln 2 - 1)\left(\ln \frac{C_i}{O_i}\right)^2 \right] \times 252$$

* **Parkinson Volatility (`park_vol_21`)** — $\sim 2.5\times$ more efficient than close-to-close variance:

  $$\sigma_P^2 = \frac{1}{n} \sum_{i=1}^n \left[ \frac{(\ln(H_i / L_i))^2}{4 \ln 2} \right] \times 252$$

#### 2.4 Momentum Indicators (4 features)
Cumulative price performance over multiple windows:

$$\text{mom\_}k\text{d} = \frac{P_t - P_{t-k}}{P_{t-k}}$$

Computed for $k \in \{5, 10, 21, 63\}$. Captures directional trend strength across horizons.

#### 2.5 Market Microstructure & Structural Risk (3 features)
* `rsi_14`: 14-day Wilder Relative Strength Index (mean-reversion pressure)
* `vol_ma_ratio`: $\frac{\text{Volume}_t}{\text{SMA}_{21}(\text{Volume})}$ (abnormal turnover spikes)
* `drawdown`: $\frac{P_t}{\max_{\tau \le t} P_\tau} - 1$ (running peak-to-trough capital decline)

---

### 3. Model Chain & Formulations

#### 3.1 GARCH(1,1) Conditional Volatility

**File:** `garch_model.py`

Asset returns have uncorrelated increments, but squared returns exhibit high autocorrelation (volatility clustering). The platform fits a standard GARCH(1,1) model:

$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0, 1)$$

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

| Parameter | Meaning | Typical Nifty 50 Value |
|---|---|---|
| $\omega$ | Baseline long-run variance weight | $\approx 0.02$ |
| $\alpha$ | Sensitivity to yesterday's shock (ARCH parameter) | $\approx 0.06$ |
| $\beta$ | Persistence of conditional variance (GARCH parameter) | $\approx 0.92$ |
| $\alpha + \beta$ | Total volatility persistence (stationarity requires $< 1$) | $\approx 0.98$ |

**Numerical Conditioning:** Returns are multiplied by 100 prior to Maximum Likelihood Estimation (`arch_model`) to ensure gradient norms remain well above machine epsilon. The fitted series is rescaled to annualized percentage volatility.

#### 3.2 3-State Gaussian Hidden Markov Model

**File:** `hmm_model.py`

Rather than using arbitrary thresholds, market regimes are discovered as latent states $z_t \in \{0, 1, 2\}$ parameterized by $\theta = (\pi, A, B)$:

* **Initial State Distribution**: $\pi_i = P(z_1 = i)$
* **Transition Matrix**: $A_{ij} = P(z_{t+1} = j \mid z_t = i)$
* **Emission Distribution**: $P(x_t \mid z_t = k) \sim \mathcal{N}(\mu_k, \Sigma_k)$ with **full covariance** $\Sigma_k$ across all 15 features.

**Monotonic State Sorting Rationale:**
HMM state indices are randomly permuted upon convergence. The platform applies post-fit relabeling by sorting state means on `garch_vol`:

$$\text{state\_order} = \text{argsort}\left( [\bar{\sigma}_{GARCH}^{(0)}, \bar{\sigma}_{GARCH}^{(1)}, \bar{\sigma}_{GARCH}^{(2)}] \right)$$

This strictly guarantees:
* State `0` = Low Volatility
* State `1` = Mid Volatility
* State `2` = High Volatility

#### 3.3 XGBoost Forward-Looking Predictive Layer

**File:** `scripts/regime_classifier.py`

The HMM is a retrospective smoother (Viterbi path across historical observations). To trade prospectively, XGBoost is trained to predict tomorrow's regime using today's feature vector:

$$\hat{y}_{t+1} = \arg\max_k P(z_{t+1} = k \mid x_t)$$

where the class probability is given by the multi-class softmax:

$$P(z_{t+1} = k \mid x_t) = \frac{\exp(F_k(x_t))}{\sum_{j=0}^2 \exp(F_j(x_t))}$$

**Walk-Forward Cross-Validation with 5-Day Gap:**
To prevent temporal leakage, validation utilizes expanding windows with a strict **5-day gap**:

```
TimeSeriesSplit(n_splits=5, gap=5)

Fold 1: Train [0..500]   ──(5-day gap)──> Validate [506..700]
Fold 2: Train [0..700]   ──(5-day gap)──> Validate [706..900]
Fold 3: Train [0..900]   ──(5-day gap)──> Validate [906..1100]
Fold 4: Train [0..1100]  ──(5-day gap)──> Validate [1106..1300]
Fold 5: Train [0..1300]  ──(5-day gap)──> Validate [1306..1500]
```

The 5-day gap prevents lookahead bias from rolling indicators (`rv_5d`, `mom_5d`) and decaying GARCH autocorrelation ($\beta \approx 0.92$).

**5-Fold Model Ensembling:**
At serving time, all 5 fold models infer probabilities on the latest feature vector:

$$\bar{P}(z_{T+1} = k \mid x_T) = \frac{1}{5} \sum_{m=1}^5 P^{(m)}(z_{T+1} = k \mid x_T)$$

#### 3.4 TreeExplainer SHAP Feature Attribution

Exact Shapley values are calculated using `shap.TreeExplainer` in polynomial time $O(TLD^2)$:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

This attributes exact contribution weights to each of the 15 features for every prediction, eliminating black-box opacity.

---

### 4. Experiment Tracking (MLflow)

**Files:** `scripts/mlflow_tracking.py`, `scripts/mlflow_sweep.py`

Three hyperparameter sweeps are permanently tracked in `mlflow.db` (SQLite backend):

| Run Name | `n_estimators` | `max_depth` | `learning_rate` | OOF Macro-AUC | Status |
|---|---|---|---|---|---|
| `sweep_n_est_200` | 200 | 4 | 0.05 | 0.7540 | Logged |
| `sweep_n_est_300` | 300 | 4 | 0.05 | 0.7540 | Logged (Production) |
| `sweep_n_est_500` | 500 | 4 | 0.05 | 0.7540 | Logged |

```bash
# Query runs directly from SQLite:
python -c "import sqlite3; conn = sqlite3.connect('mlflow.db'); [print(r) for r in conn.cursor().execute('SELECT name, status FROM runs').fetchall()]"

# Or launch the interactive MLflow UI:
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

### 5. Backtesting Framework

**File:** `backtest.py`

The adaptive regime-switching strategy rebalances exposure based on the detected state:

| Regime | Sizing | Action | Rationale |
|---|---:|---|---|
| **0 (Low Vol)** | 1.0× | Full Long | Maximize participation in calm trending markets |
| **1 (Mid Vol)** | 0.5× | Half Long | De-risk portfolio during uncertain consolidation |
| **2 (High Vol)** | 0.0× | 100% Cash | Full capital preservation during market crises |

#### Realistic Execution Model:
* **Initial Capital**: ₹1,00,000
* **Execution Lag**: 1 day (signal computed on Day $T$ close $\rightarrow$ executed at Day $T+1$ open)
* **Transaction Fees**: 0.10% per transaction (brokerage + STT + stamp duty)
* **Slippage**: 0.10% per execution (bid-ask spread & market impact)
* **Risk-free Rate**: 6.50% (India 10-Year G-Sec benchmark)

---

### 6. Serving Layer

#### FastAPI Backend (`app/main.py`)
* Serves data, models, and predictions via 7 REST endpoints.
* Employs `@lru_cache(maxsize=1)` to retain models and precomputed features in memory, delivering sub-10ms response latencies.

#### Next.js Frontend (`frontend/`)
* Modern dark-themed dashboard built with Next.js 14 and Tailwind CSS.
* Visualizes real-time metrics, probability dials, SHAP attribution bars, and historical regime classification logs.
* Features automatic cold-start retry logic for Render free-tier instances.

#### External Integrations
* Directly consumed by the [SEBI RAG Bot](https://github.com/RaajitSingh1306/sebi-rag-bot)'s `quant_agent` via HTTP to provide live quantitative context alongside legal compliance queries.

---

## Where

### Project Directory Layout

```
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
├── outputs/
│   ├── backtest.png             # Cumulative returns chart
│   └── shap_summary_xgb.png     # SHAP feature importance plot
├── research/
│   ├── 01_arima_null_result.ipynb          # ADF test & baseline null result
│   └── 02_sector_momentum_preview.ipynb    # Cross-sector volatility preview
├── results/
│   └── metrics.json             # Machine-readable evaluation scores
├── scripts/
│   ├── regime_classifier.py     # Walk-forward CV & XGBoost training
│   ├── mlflow_sweep.py          # MLflow hyperparameter sweep script
│   └── mlflow_tracking.py       # MLflow logging utilities
├── tests/
│   ├── test_api.py              # 8 API endpoint tests
│   ├── test_backtest.py         # 4 backtest logic tests
│   ├── test_features.py         # 4 feature calculation tests
│   ├── test_garch.py            # 2 GARCH convergence tests
│   ├── test_hmm.py              # 3 HMM sorting and regime tests
│   └── test_models.py           # 3 XGBoost inference tests
├── backtest.py                  # vectorbt backtesting engine
├── data.py                      # Data ingestion & caching layer
├── features.py                  # 15 volatility/momentum features
├── garch_model.py               # GARCH(1,1) model implementation
├── hmm_model.py                 # 3-state Gaussian HMM implementation
├── run.py                       # Pipeline runner (GARCH + HMM + Backtest)
├── Dockerfile                   # Production multi-stage Docker build
├── render.yaml                  # Render deployment blueprint
└── requirements.txt             # Production Python dependencies
```

---

### REST API Reference

All endpoints return standard JSON and include interactive OpenAPI documentation at `/docs`.

| Endpoint | Method | Response Description |
|---|:---:|---|
| `/` | `GET` | Health check and system verification |
| `/current` | `GET` | Current Nifty 50 close, regime label, and annualized GARCH vol |
| `/predict` | `GET` | Next-day XGBoost ensemble probability distribution (`prob_low`, `prob_mid`, `prob_high`) |
| `/market-summary` | `GET` | Aggregated KPIs: current regime, Sharpe ratio, max drawdown, OOF AUC |
| `/regimes?last_n=252` | `GET` | Time-series history of daily closes, regimes, and conditional volatilities |
| `/stats` | `GET` | Per-regime summary metrics (mean return, annualized volatility, frequency) |
| `/backtest-summary` | `GET` | Comparative backtest statistics: strategy vs Buy & Hold |

#### Example: `/predict`

```bash
curl -X GET https://volatility-intelligence-platform.onrender.com/predict
```

```json
{
  "predicted_regime": 0,
  "predicted_regime_name": "Low Vol",
  "probability_distribution": {
    "Low Vol": 0.6582,
    "Mid Vol": 0.2311,
    "High Vol": 0.1107
  },
  "model_oof_macro_auc": 0.7453,
  "data_end_date": "2026-09-16"
}
```

---

### Getting Started

#### Prerequisites
* Python 3.11+
* Node.js 18+ (for frontend)
* Docker (optional, recommended for production)

#### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/RaajitSingh1306/volatility-intelligence-platform.git
cd volatility-intelligence-platform

# 2. Set up Python virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the quantitative pipeline
python run.py

# 5. Train the XGBoost predictive ensemble
python scripts/regime_classifier.py

# 6. Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 7. (Optional) Run the Next.js frontend
cd frontend
npm install
npm run dev
```

#### Docker (Recommended)

```bash
# Build and run container
docker build -t volatility-platform .
docker run -p 8000:8000 volatility-platform
```

#### Testing

```bash
# Run complete test suite (24 tests)
pytest tests/ -v
```

---

### Evaluation & Benchmarks

#### Predictive Model: XGBoost Walk-Forward CV

* Target Metric: Out-of-fold Macro-AUC $> 0.70$
* Achieved Score: **0.7453** ✅

| Fold | Training Window | Validation Window | Fold Macro-AUC | Notes |
|:---:|:---:|:---:|:---:|---|
| **Fold 1** | Days 0–500 | Days 506–700 | 0.5000 | Limited training data in early window |
| **Fold 2** | Days 0–700 | Days 706–900 | 0.9094 | Strong classification across regime boundaries |
| **Fold 3** | Days 0–900 | Days 906–1100 | 0.9844 | Excellent identification of transition signals |
| **Fold 4** | Days 0–1100 | Days 1106–1300 | 0.5000 | Prolonged single-regime bull market |
| **Fold 5** | Days 0–1300 | Days 1306–1500 | 0.8584 | Strong detection of elevated volatility periods |
| **Overall** | **Expanding** | **Walk-Forward** | **0.7453** | **Production Ensemble Macro-AUC** |

#### Backtest Performance (2014 – Present, Nifty 50)

* Benchmark: Nifty 50 Buy & Hold
* Hurdle Rate: 6.50% (India 10-Year Government Bond)
* Friction: 0.10% fees + 0.10% slippage + 1-day execution lag

| Metric | Adaptive Regime Strategy | Buy & Hold Benchmark | Alpha / Difference |
|---|---:|---:|:---:|
| **Sharpe Ratio** | **0.968** | 0.873 | **+0.095** |
| **Max Drawdown** | **-25.92%** | -38.44% | **+12.52% capital protection** |
| **Total Return** | **+262.01%** | +249.48% | **+12.53%** |
| **CAGR** | **16.61%** | 16.13% | **+0.48%** |
| **Calmar Ratio** | **0.641** | 0.419 | **+0.222** |

#### Serving Latency

| Operation | Latency |
|---|---|
| First request (Cold load from disk) | $\sim 2$ seconds |
| Subsequent requests (LRU cached) | $< 10$ ms |
| `/predict` 5-fold ensemble inference | $< 50$ ms |

---

### Deployment

#### Backend → Render

```yaml
# render.yaml blueprint
services:
  - type: web
    name: volatility-intelligence-platform
    runtime: docker
    dockerfilePath: Dockerfile
    plan: free
    healthCheckPath: /
    envVars:
      - key: PORT
        value: 8000
```

1. Connect repository in Render Dashboard.
2. Select **Blueprint** to deploy using `render.yaml`.
3. Health check verifies `/` status upon build completion.

#### Frontend → Vercel
1. Import repository on [Vercel](https://vercel.com).
2. Set root directory to `frontend`.
3. Configure environment variable: `NEXT_PUBLIC_API_URL=https://volatility-intelligence-platform.onrender.com`.
4. Deploy.

---

### Connected Projects

This platform serves as the flagship quantitative foundation for an interconnected quantitative finance and engineering ecosystem:

| Project | Domain / Role | GitHub Repository |
|---|---|---|
| **Volatility Intelligence Platform** (This Repo) | Flagship GARCH + HMM + XGBoost production prediction platform | [volatility-intelligence-platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform) |
| **SEBI RAG Bot** | Multi-agent compliance assistant consuming this volatility API | [sebi-rag-bot](https://github.com/RaajitSingh1306/sebi-rag-bot) |
| **NSEI Daily Stock Pipeline** | Lakehouse & ML feature store materializing rolling metrics | [NSEI-Daily-Stock-Pipeline](https://github.com/RaajitSingh1306/NSEI-Daily-Stock-Pipeline) |
| **Nifty Sector Rotation** | Momentum strategy utilizing dynamic regime volatility filters | [Nifty-Sector-Rotation](https://github.com/RaajitSingh1306/Nifty-Sector-Rotation) |
| **Nifty Time Series** | Empirical research proving daily return unpredictability (EMH baseline) | [Nifty-Time-Series](https://github.com/RaajitSingh1306/Nifty-Time-Series) |
| **Credit Default Predictor** | Loan default prediction with TreeSHAP feature attribution | [Credit-Default-Predictor](https://github.com/RaajitSingh1306/Credit-Default-Predictor) |
| **Global Market HeatMap** | Cross-border 35-stock risk-return analytics & dashboard | [Global-Equity-Market-Dashboard](https://github.com/RaajitSingh1306/Global-Equity-Market-Dashboard) |
| **Finance KPI** | Automated portfolio performance & drawdown diagnostics | [Finance_Kpi](https://github.com/RaajitSingh1306/Finance_Kpi) |
| **Volatility Classifier (Simplified)** | *Superseded (v2)*: Single-asset 15-feature GARCH + HMM refactor | [volatility-classifier-simplified](https://github.com/RaajitSingh1306/volatility-classifier-simplified) |
| **Volatility Regime Classifier** | *Superseded (v1)*: Dual-asset 3-feature HMM prototype | [Volatility-Regime-Classifier](https://github.com/RaajitSingh1306/Volatility-Regime-Classifier) |

---

## Limitations & Roadmap

### Known Limitations
- **Hardcoded HMM Regimes**: The model fixes the hidden state count at $K=3$ (Low, Medium, High). It does not dynamically calibrate state count via BIC/AIC criterion sweeps during market transitions.
- **Symmetric Volatility Clustering**: Relies on standard GARCH(1,1) with a normal distribution, ignoring asymmetric leverage effects (where negative shocks generate higher volatility than equivalent positive shocks) modeled by EGARCH or GJR-GARCH.
- **Static Ingestion & Caching**: Data ingestion relies on yfinance with a 7-day TTL file cache (`data.py`) rather than low-latency real-time tick/websocket streaming feeds.
- **No Online Learning**: The XGBoost ensemble uses 5-fold cross-validated bagging; it requires batch retraining rather than incremental/online continuous learning.
- **Simplified Transaction Cost Model**: The backtesting module applies flat 0.10% round-trip trading friction, omitting exchange turnover charges, Securities Transaction Tax (STT) slabs, slippage distributions, or market impact at institutional scale.
- **Walk-Forward Regime Dominance**: In walk-forward splits dominated by a single regime (e.g. Folds 1 and 4), direction accuracy degraded to random-chance (~0.5000), highlighting the need for fold-specific dynamic thresholding.
- **Cloud Free-Tier Cold Starts**: Deployment on Render free tier introduces a 30–60 second container spin-up latency on initial invocation.

### Roadmap
- [ ] **Asymmetric Volatility Modeling**: Implement EGARCH(1,1) and GJR-GARCH with Student's $t$ innovations to capture heavy tails and leverage asymmetry.
- [ ] **Automated Retraining Loop**: Connect directly to upstream triggers from `NSEI Daily Stock Pipeline` for automated daily/weekly retraining and MLflow model registry updates.
- [ ] **Fold-Aware Adaptive Thresholding**: Implement dynamic probability thresholds per regime to stabilize walk-forward classification on imbalanced folds.
- [ ] **Multi-Asset Expansion**: Extend the multi-step regime and volatility forecasting architecture to Bank Nifty (`^NSEBANK`) and global benchmarks (S&P 500, Nasdaq 100).
- [ ] **Real-Time Notification Engine**: Add Slack and Telegram webhook alerts triggered upon high-volatility regime state transitions.

---

## License & Disclaimer

MIT License. This project is developed strictly for **educational and scientific research purposes** and does not constitute financial, investment, or trading advice. Past performance under backtested simulation is not indicative of future returns.

