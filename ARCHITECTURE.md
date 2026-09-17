# Volatility Intelligence Platform — System Architecture

> In-depth architectural documentation, mathematical foundations, model chaining justifications, and design rationale for the Nifty 50 Volatility Intelligence Platform.

---

## Table of Contents

1. [End-to-End System Architecture](#1-end-to-end-system-architecture)
2. [Data Layer](#2-data-layer)
3. [Feature Engineering Layer](#3-feature-engineering-layer)
4. [Model Chain & Mathematical Foundations](#4-model-chain--mathematical-foundations)
   - 4.1 [GARCH(1,1) Conditional Volatility](#41-garch11-conditional-volatility)
   - 4.2 [Hidden Markov Model Regime Detection](#42-hidden-markov-model-regime-detection)
   - 4.3 [XGBoost Predictive Layer](#43-xgboost-predictive-layer)
5. [Experiment Tracking (MLflow)](#5-experiment-tracking-mlflow)
6. [Backtesting Framework](#6-backtesting-framework)
7. [Serving Layer](#7-serving-layer)
8. [Design Decisions & Trade-offs](#8-design-decisions--trade-offs)
9. [Performance Benchmarks](#9-performance-benchmarks)

---

## 1. End-to-End System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Data Layer                              │
│   Yahoo Finance (yfinance) → data.py → data/nifty_ohlcv.csv     │
│   7-day local TTL cache. Re-downloads when stale.                │
└─────────────────────┬───────────────────────────────────────────┘
                      │ OHLCV: Date, Open, High, Low, Close, Volume
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Feature Engineering Layer                       │
│   features.py → 15 features across 5 categories                 │
│   Realized vol · Vol-of-vol · OHLC estimators · Momentum · Micro│
└────────────────┬────────────────────┬───────────────────────────┘
                 │                    │
                 ▼                    ▼
┌──────────────────────┐   ┌──────────────────────────────────────┐
│   GARCH(1,1) Layer   │   │            HMM Layer                  │
│   garch_model.py     │   │   hmm_model.py                        │
│                      │   │                                        │
│   σ²ₜ = ω + αε²ₜ₋₁  │   │   3-state Gaussian, full covariance   │
│         + βσ²ₜ₋₁    │   │   Sorted by ascending mean garch_vol  │
│                      │   │   → 0=Low, 1=Mid, 2=High              │
│   → garch_vol series │   │   → regime labels (retrospective)     │
└──────────────────────┘   └──────────────┬───────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   XGBoost Prediction Layer                       │
│   scripts/regime_classifier.py                                   │
│                                                                  │
│   Input:   X = 15 features on day T                              │
│   Target:  y = HMM regime label on day T+1 (TOMORROW)            │
│   CV:      TimeSeriesSplit(n_splits=5, gap=5)                    │
│   Model:   XGBClassifier(multi:softprob, 3 classes)              │
│   Explain: TreeExplainer SHAP (exact Shapley values)             │
│                                                                  │
│   Artifacts:                                                     │
│     → models/xgb_bundle.pkl  (5 fold models + scaler + OOF AUC) │
│     → outputs/shap_summary_xgb.png                               │
│     → results/metrics.json                                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
┌──────────────────────┐   ┌──────────────────────────────────────┐
│   MLflow Tracking    │   │         vectorbt Backtest             │
│                      │   │   backtest.py                         │
│   mlflow.db (SQLite) │   │                                       │
│   3 sweeps logged:   │   │   Position sizing by regime:          │
│   n_est ∈ {200,300,  │   │     0 (Low)  → 1.0× Long             │
│            500}      │   │     1 (Mid)  → 0.5× Long             │
│   Params, fold AUCs, │   │     2 (High) → 0.0× Cash             │
│   OOF AUC, artifacts │   │   Execution: 1-day lag, 0.10% fees   │
│                      │   │                                       │
│   → mlruns/          │   │   → data/backtest_summary.csv         │
│   → mlflow.db        │   │   → outputs/backtest.png              │
└──────────────────────┘   └──────────────────┬───────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Serving Layer                            │
│                                                                  │
│   FastAPI (app/main.py)            Next.js (frontend/)           │
│   Render / Docker :8000            Vercel :3000                  │
│                                                                  │
│   GET /current                     Hero KPI cards                │
│   GET /predict                     Regime probability gauges     │
│   GET /market-summary              SHAP feature breakdown        │
│   GET /regimes?last_n=252          Regime timeline chart          │
│   GET /stats                       Per-regime statistics table   │
│   GET /backtest-summary            Strategy equity curve          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Layer

**File:** `data.py`

| Aspect | Detail |
|---|---|
| **Source** | Yahoo Finance via `yfinance` — ticker `^NSEI` (Nifty 50 Index) |
| **Period** | 2014-01-01 to present (~2,700+ trading days) |
| **Fields** | Date, Open, High, Low, Close, Volume (OHLCV) |
| **Derived** | `returns` = daily percentage change, `log_returns` = ln(Pₜ/Pₜ₋₁) |
| **Caching** | Local CSV with 7-day TTL. If `nifty_ohlcv.csv` is < 7 days old, skips download. |

**Why 2014?** Captures multiple market regimes: the 2015-16 China devaluation selloff, 2018 IL&FS crisis, COVID-19 crash (March 2020), and the post-pandemic bull run. This diversity ensures the HMM has sufficient samples of all three regime types.

---

## 3. Feature Engineering Layer

**File:** `features.py`

15 features organized into 5 categories:

### 3.1 Realized Volatility (4 features)

Rolling standard deviation of log returns over multiple horizons:

$$RV_n = \text{std}(r_{t-n+1}, \ldots, r_t) \times \sqrt{252}$$

| Feature | Window | Purpose |
|---|---|---|
| `rv_5d` | 5 days | Very short-term volatility (weekly) |
| `rv_10d` | 10 days | Short-term volatility (biweekly) |
| `rv_21d` | 21 days | Monthly volatility (standard measure) |
| `rv_63d` | 63 days | Quarterly volatility (structural) |

### 3.2 Volatility-of-Volatility (2 features)

Ratios of short-to-long realized volatility:

| Feature | Formula | Purpose |
|---|---|---|
| `vol_ratio_5_21` | `rv_5d / rv_21d` | Detects sudden volatility spikes (value > 1 = acceleration) |
| `vol_ratio_21_63` | `rv_21d / rv_63d` | Detects sustained volatility regime shifts |

### 3.3 OHLC Range-Based Estimators (2 features)

More efficient than close-to-close volatility because they use intraday price information:

| Feature | Estimator | Formula Intuition |
|---|---|---|
| `gk_vol_21` | **Garman-Klass** | Uses Open, High, Low, Close — ~5× more efficient than close-to-close |
| `park_vol_21` | **Parkinson** | Uses High-Low range — ~2.5× more efficient than close-to-close |

### 3.4 Momentum (4 features)

Cumulative returns over multiple windows:

| Feature | Window | Purpose |
|---|---|---|
| `mom_5d` | 5 days | Short-term trend |
| `mom_10d` | 10 days | Biweekly trend |
| `mom_21d` | 21 days | Monthly trend |
| `mom_63d` | 63 days | Quarterly trend — captures cyclical patterns |

### 3.5 Market Microstructure (3 features)

| Feature | Definition | Purpose |
|---|---|---|
| `rsi_14` | 14-day Relative Strength Index | Overbought/oversold — captures mean-reversion pressure |
| `vol_ma_ratio` | Volume / 21-day MA Volume | Abnormal volume detection — spikes often precede regime transitions |
| `drawdown` | Current price / rolling max − 1 | Peak-to-trough decline — quantifies ongoing capital loss |

---

## 4. Model Chain & Mathematical Foundations

### 4.1 GARCH(1,1) Conditional Volatility

**File:** `garch_model.py`

Daily returns exhibit near-zero autocorrelation (consistent with weak-form EMH), but squared returns show strong persistence — the **ARCH effect**. GARCH(1,1) captures this:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

| Parameter | Meaning | Typical Nifty 50 Value |
|---|---|---|
| ω | Long-run baseline variance | ~0.02 |
| α | Yesterday's shock impact | ~0.06 |
| β | Volatility persistence/memory | ~0.92 |
| α + β | Stationarity check (must be < 1) | ~0.98 |

**Why α + β ≈ 0.98 matters:** This is very close to 1, indicating **long-memory volatility clustering**. A volatile day is likely followed by more volatile days. This persistence is exactly what makes regime detection useful — regimes are sticky, not random.

**Numerical conditioning:** Raw returns (~0.01 magnitude) produce poorly scaled variances (~0.0001). Scaling returns by 100 before estimation improves the maximum likelihood optimizer's convergence. The output is then rescaled back to annualized percentage volatility.

### 4.2 Hidden Markov Model Regime Detection

**File:** `hmm_model.py`

Rather than defining volatility thresholds manually (e.g., "above 20% = high vol"), the HMM discovers three latent market states through unsupervised learning:

$$\theta = (\pi, A, B)$$

| Component | Definition | Role |
|---|---|---|
| π | Initial state distribution | Prior probability of starting in each state |
| A | Transition matrix (3×3) | $A_{ij} = P(z_t = j \mid z_{t-1} = i)$ — regime persistence and switching rates |
| B | Emission distributions | $P(x_t \mid z_t = k) \sim \mathcal{N}(\mu_k, \Sigma_k)$ — full covariance over 15 features |

**Full covariance:** Each state has its own 15×15 covariance matrix, capturing feature correlations within each regime. For example, during High Vol states, realized volatility and drawdown are strongly correlated — this joint structure helps the HMM distinguish regimes that differ in their correlation patterns, not just their means.

**State sorting problem:** HMM state indices are arbitrary — `state 0` might be High Vol in one training run and Low Vol in another. The pipeline resolves this by sorting states by **ascending mean GARCH volatility** after fitting:

```
state_order = np.argsort([features_in_state_k['garch_vol'].mean() for k in range(3)])
label_map = {old: new for new, old in enumerate(state_order)}
```

This guarantees `0 = Low Vol`, `1 = Mid Vol`, `2 = High Vol` across all runs, regardless of random initialization.

### 4.3 XGBoost Predictive Layer

**File:** `scripts/regime_classifier.py`

The HMM assigns regime labels **retrospectively** — it uses the Viterbi algorithm on the entire observed sequence. For trading, we need **prospective** predictions: what regime will the market be in _tomorrow_?

XGBoost solves this by learning:

$$P(z_{t+1} = k \mid x_t) = \frac{e^{F_k(x_t)}}{\sum_{j=0}^2 e^{F_j(x_t)}}$$

where $F_k$ is the gradient-boosted tree ensemble for class $k$.

**Walk-forward cross-validation:**

Standard k-fold CV is invalid for time series because it shuffles data, letting the model see future observations during training. Walk-forward expanding window preserves temporal order:

```
TimeSeriesSplit(n_splits=5, gap=5)

Fold 1: Train [0..500]        → Validate [506..700]
Fold 2: Train [0..700]        → Validate [706..900]
Fold 3: Train [0..900]        → Validate [906..1100]
Fold 4: Train [0..1100]       → Validate [1106..1300]
Fold 5: Train [0..1300]       → Validate [1306..1500]
```

**The 5-day gap:** Each fold leaves a **5-day buffer** between training and validation sets. This prevents autoregressive leakage from:
- Rolling features (`rv_5d` uses the last 5 days — shared across the gap)
- GARCH memory (β ≈ 0.92 means volatility estimates carry ~5 days of meaningful information)

**Ensemble inference:** At serving time, all 5 fold models predict on the latest feature row. Their probability vectors are averaged, producing smoother and more calibrated estimates than any single fold model.

**SHAP explainability:**

`shap.TreeExplainer` computes **exact Shapley values** through direct tree traversal in $O(TLD^2)$ time (T = trees, L = leaves, D = depth). This is fundamentally different from `KernelExplainer` which uses $O(2^N)$ sampling approximations and can produce inconsistent attributions.

---

## 5. Experiment Tracking (MLflow)

**Files:** `scripts/mlflow_tracking.py`, `scripts/mlflow_sweep.py`

Three hyperparameter sweeps are permanently logged in `mlflow.db` (SQLite backend):

| Run Name | `n_estimators` | `max_depth` | `learning_rate` | OOF Macro-AUC |
|---|---|---|---|---|
| `sweep_n_est_200` | 200 | 4 | 0.05 | 0.7540 |
| `sweep_n_est_300` | 300 | 4 | 0.05 | 0.7540 |
| `sweep_n_est_500` | 500 | 4 | 0.05 | 0.7540 |

**What's logged per run:**
- All hyperparameters (n_estimators, max_depth, learning_rate, n_splits, gap)
- Per-fold AUC scores (5 values)
- Aggregate OOF macro-AUC
- Model bundle artifact path

**Observation:** AUC is stable across `n_estimators ∈ {200, 300, 500}` — the model saturates early, suggesting the signal-to-noise boundary is reached by 200 trees. The production bundle uses 300 trees as a balanced choice.

**Verification:**
```bash
# Quick CLI check
python -c "import sqlite3; conn = sqlite3.connect('mlflow.db'); [print(f'Run: {r[1]} | Status: {r[2]}') for r in conn.cursor().execute('SELECT run_uuid, name, status FROM runs').fetchall()]"

# Full web UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## 6. Backtesting Framework

**File:** `backtest.py`

### Strategy Rules

| Regime | Position Size | Rationale |
|---|---|---|
| 0 (Low Vol) | **1.0× Long** | Favorable risk environment — full exposure |
| 1 (Mid Vol) | **0.5× Long** | Uncertain — reduce exposure |
| 2 (High Vol) | **0.0× Cash** | Capital preservation — exit entirely |

### Execution Model

| Parameter | Value | Why |
|---|---|---|
| **Initial capital** | ₹1,00,000 | Standard backtest baseline |
| **Execution lag** | 1 day | Signal at day T close → execute at day T+1 open. Prevents forward bias. |
| **Fees** | 0.1% per trade | Covers brokerage + STT + exchange charges (Indian market realistic) |
| **Slippage** | 0.1% | Accounts for bid-ask spread and market impact on Nifty ETF/futures |
| **Benchmark** | Nifty 50 Buy & Hold | Full-period passive investment |

### Why These Parameters Matter

Many published backtests omit fees, slippage, and execution lag — producing artificially inflated Sharpe ratios. This backtest explicitly includes all three to ensure the reported alpha is achievable in practice.

---

## 7. Serving Layer

### FastAPI Backend (`app/main.py`)

| Endpoint | Method | Response Schema | Purpose |
|---|---|---|---|
| `/` | GET | `{"status": "ok", "message": "..."}` | Health check and API identification |
| `/current` | GET | `CurrentRegime(date, regime, regime_name, garch_vol_pct, close)` | Latest regime classification |
| `/predict` | GET | `PredictResponse(predicted_regime, prob_low/mid/high, model_auc)` | Next-day ensemble prediction |
| `/market-summary` | GET | `MarketSummaryResponse(regime, vol, close, sharpe, max_dd, auc)` | Dashboard hero KPIs |
| `/regimes?last_n=N` | GET | `list[RegimePoint]` | Historical regime time series |
| `/stats` | GET | `list[dict]` | Per-regime aggregate statistics |
| `/backtest-summary` | GET | `dict` | Strategy vs B&H comparison table |

**Caching strategy:** Data and model loaders use `@lru_cache(maxsize=1)` — the CSV and pickle files load once at first request, then serve from memory. This provides **millisecond response times** with zero allocation overhead.

### Next.js Frontend (`frontend/`)

- Built with Next.js 14 + Tailwind CSS (dark theme)
- Fetches from all 7 API endpoints on page load
- Auto-reconnects when Render wakes from cold start (free tier spins down after ~15 min inactivity)

### External Consumer

The [SEBI RAG Bot](https://github.com/RaajitSingh1306/sebi-rag-bot)'s `quant_agent` calls `/current` and `/stats` via `httpx` to answer market volatility questions within the compliance chat interface.

---

## 8. Design Decisions & Trade-offs

| Decision | Implementation | Rationale |
|---|---|---|
| **HMM before XGBoost** | HMM labels → XGBoost targets | HMM discovers regimes unsupervised; XGBoost learns the temporal mapping to predict them forward |
| **Gap in CV** | `TimeSeriesSplit(gap=5)` | Prevents GARCH lag leakage. Without it, the last 5 training rows contaminate the first 5 validation rows through rolling features |
| **Ensemble averaging** | Mean of 5 fold predictions | Smooths out-of-fold variance. Any single fold model may overfit to its training window |
| **1-day execution lag** | Signal at T → trade at T+1 | Eliminates forward-looking bias. Many backtests implicitly assume same-day execution, which is unrealistic |
| **TreeExplainer** | `shap.TreeExplainer` (not Kernel) | Exact Shapley values in O(TLD²) vs exponential approximation. Deterministic and consistent |
| **Full covariance HMM** | `covariance_type='full'` | Captures regime-specific feature correlations (e.g., vol-drawdown correlation strengthens in High Vol). Diagonal covariance would miss this |
| **State sorting** | Sort by ascending mean `garch_vol` | Removes HMM's inherent state-index ambiguity. Labels are semantically consistent across runs |
| **LRU caching** | `@lru_cache(maxsize=1)` | Single-alloc data/model load. Subsequent API requests serve from memory with zero I/O |
| **CSV/PKL contracts** | No external database | Each pipeline step reads and writes flat files. Components can run independently, simplifying debugging and CI testing |
| **Docker build-time training** | `RUN python run.py && python scripts/regime_classifier.py` | Model artifacts baked into image. Container starts instantly with no runtime data dependencies |
| **Log return scaling** | `returns × 100` before GARCH | Well-conditioned MLE optimization. Raw returns (~0.01) produce variance estimates near machine epsilon |

---

## 9. Performance Benchmarks

### Predictive Model (XGBoost Walk-Forward CV)

| Metric | Value | Target | Status |
|---|---|---|:---:|
| **OOF Macro-AUC** | **0.7453** | > 0.70 | ✅ PASS |
| Fold 1 AUC | 0.5000 | — | Monitored (limited training data) |
| Fold 2 AUC | 0.9094 | — | Strong |
| Fold 3 AUC | 0.9844 | — | Excellent |
| Fold 4 AUC | 0.5000 | — | Monitored (homogeneous regime period) |
| Fold 5 AUC | 0.8584 | — | Strong |

### Backtest (2014 – Present, Nifty 50, Rf = 6.5%)

| Metric | Regime Strategy | Buy & Hold | Alpha |
|---|---|---|---|
| **Sharpe Ratio** | **0.968** | 0.873 | +0.095 |
| **Max Drawdown** | **-25.92%** | -38.44% | **+12.52% capital protection** |
| **Total Return** | +262.01% | +249.48% | +12.53% |
| **CAGR** | 16.61% | 16.13% | +0.48% |
| **Calmar Ratio** | **0.641** | 0.419 | +0.222 |

### API Performance

| Metric | Value |
|---|---|
| Cold start (first request, loads data + model) | ~2-3 seconds |
| Warm request (cached) | < 10 ms |
| `/predict` ensemble (5 models) | < 50 ms |
| Container build time (Docker) | ~3-5 minutes |
