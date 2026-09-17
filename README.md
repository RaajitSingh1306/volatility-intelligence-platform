# Volatility Intelligence Platform — Nifty 50

[![CI Tests](https://img.shields.io/badge/tests-24%2F24%20passed-brightgreen)](#testing)
[![OOF Macro-AUC](https://img.shields.io/badge/OOF%20Macro--AUC-0.7453-blue)](#evaluation--benchmarks)
[![Sharpe Ratio](https://img.shields.io/badge/Sharpe-0.968%20vs%200.873%20B%26H-emerald)](#evaluation--benchmarks)
[![Docker](https://img.shields.io/badge/Docker-Verified-cyan)](#getting-started)
[![Deploy on Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7)](#deployment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

| | |
|---|---|
| **Web Dashboard** | https://volatility-intelligence-platform.vercel.app |
| **REST API** | https://volatility-intelligence-platform.onrender.com |
| **API Docs** | https://volatility-intelligence-platform.onrender.com/docs |

---

## What

The Volatility Intelligence Platform is a **production-grade quantitative system** that classifies and predicts Nifty 50 volatility regimes. It chains four models into a single pipeline:

1. **GARCH(1,1)** estimates daily conditional volatility
2. **3-state Gaussian HMM** discovers latent market regimes (Low / Mid / High Vol)
3. **XGBoost ensemble** predicts tomorrow's regime from today's features
4. **vectorbt backtest** validates the strategy against Nifty 50 Buy & Hold

The platform serves predictions through a **FastAPI REST API** with 7 endpoints and a **Next.js dashboard** showing real-time KPIs, regime probabilities, SHAP explanations, and backtest equity curves.

### What It Produces

| Output | Description |
|---|---|
| **Current regime** | Today's volatility state (Low / Mid / High) based on HMM classification |
| **Next-day prediction** | XGBoost ensemble probability distribution over tomorrow's regime |
| **GARCH volatility** | Annualized conditional volatility estimate updated daily |
| **SHAP explanations** | Feature importance showing which indicators drive the prediction |
| **Backtest report** | Sharpe ratio, max drawdown, CAGR comparison vs Buy & Hold |
| **MLflow experiment log** | 3 hyperparameter sweeps with full parameter and metric tracking |

---

## Why

### The Problem

Institutional quant desks have access to expensive Bloomberg terminals and proprietary regime models. Retail investors and independent researchers have none of this infrastructure. The gap manifests as:

1. **No objective volatility measurement** — "the market feels risky" is not a tradeable signal
2. **No forward-looking prediction** — GARCH and HMM tell you what regime you're in _today_, not what's coming _tomorrow_
3. **No validation** — most "quant strategies" on GitHub have no walk-forward cross-validation, no leakage controls, and no realistic backtest with fees and slippage
4. **No experiment tracking** — hyperparameter choices are undocumented, making results irreproducible
5. **No production API** — models exist as notebooks, not as deployable services that other systems can consume

### The Solution

This platform addresses every gap:

| Problem | Solution |
|---|---|
| No volatility measurement | **GARCH(1,1)** provides a time-varying conditional volatility estimate |
| Only backward-looking | **XGBoost** learns to predict tomorrow's HMM regime from today's 15 features |
| No validation rigor | **Walk-forward CV** with `TimeSeriesSplit(n_splits=5, gap=5)` — no lookahead leakage |
| No experiment tracking | **3 MLflow runs** logged in `mlflow.db` with params, fold AUCs, and artifacts |
| No production API | **FastAPI** with 7 endpoints + **Next.js** dashboard + **Render** cloud deployment |
| No realistic backtest | **vectorbt** with 0.1% fees, 0.1% slippage, 1-day execution lag |

### How This Differs From the Simplified Version

| Feature | Simplified | Intelligence Platform (this repo) |
|---|---|---|
| Regime classification | HMM only | HMM + XGBoost prediction |
| Next-day forecast | ❌ | ✅ `/predict` endpoint |
| MLflow tracking | ❌ | ✅ 3 logged sweeps |
| Frontend | Streamlit | Next.js (production dark-theme dashboard) |
| API endpoints | 5 | 7 (adds `/predict` and `/market-summary`) |
| Test count | 12 | 24 |
| Deployment | Python runtime | Docker container (trains during build) |

---

## How

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Layer (yfinance)                       │
│   data.py → data/nifty_ohlcv.csv (7-day cache TTL)             │
└─────────────────────┬───────────────────────────────────────────┘
                      │ OHLCV: Date, Open, High, Low, Close, Volume
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Feature Engineering Layer                      │
│   features.py → 15 features across 5 categories                │
└────────────────┬────────────────────┬───────────────────────────┘
                 │                    │
                 ▼                    ▼
┌──────────────────────┐   ┌──────────────────────────────────────┐
│   GARCH(1,1) Layer   │   │            HMM Layer                 │
│   garch_model.py     │   │   hmm_model.py                       │
│   Conditional vol    │   │   Gaussian HMM, 3 states, full cov   │
│   → garch_vol series │   │   Sorted by ascending mean vol       │
└──────────────────────┘   │   → 0=Low, 1=Mid, 2=High             │
                           └──────────────┬───────────────────────┘
                                          │ HMM labels
                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   XGBoost Prediction Layer                      │
│   scripts/regime_classifier.py                                  │
│                                                                 │
│   Input:  X = 15 features on day T                              │
│   Target: y = HMM regime label on day T+1 (tomorrow)            │
│   CV:     TimeSeriesSplit(n_splits=5, gap=5)                    │
│   Model:  XGBClassifier(multi:softprob, 3 classes)              │
│   SHAP:   TreeExplainer exact attributions                      │
│   Output: models/xgb_bundle.pkl (5 fold models + OOF AUC)      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
┌──────────────────────┐   ┌──────────────────────────────────────┐
│   MLflow Tracking    │   │         vectorbt Backtest            │
│   mlruns/ (3 sweeps) │   │   Enter on Low Vol, exit on High Vol │
│   Params + fold AUCs │   │   1-day lag, 0.1% fees + slippage    │
└──────────────────────┘   └──────────────────┬───────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Serving Layer                           │
│   FastAPI (Render :8000)    ⇄    Next.js (Vercel :3000)         │
│   7 endpoints                    Real-time KPI dashboard        │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Step-by-Step

#### Step 1: Data Ingestion (`data.py`)

Downloads Nifty 50 OHLCV from Yahoo Finance (`^NSEI`, 2014-01-01 to present). Implements 7-day local cache to avoid redundant downloads. Computes daily returns and log returns.

#### Step 2: Feature Engineering (`features.py`)

Generates 15 features from raw OHLCV:

| Category | Features | Purpose |
|---|---|---|
| Realized Volatility | `rv_5d`, `rv_10d`, `rv_21d`, `rv_63d` | Rolling standard deviation of log returns at multiple horizons |
| Vol-of-Volatility | `vol_ratio_5_21`, `vol_ratio_21_63` | Short/long vol ratios — detect volatility acceleration |
| OHLC Estimators | `gk_vol_21` (Garman-Klass), `park_vol_21` (Parkinson) | Intraday range-based volatility — more efficient than close-to-close |
| Momentum | `mom_5d`, `mom_10d`, `mom_21d`, `mom_63d` | Cumulative returns over multiple windows |
| Microstructure | `rsi_14`, `vol_ma_ratio`, `drawdown` | Relative strength, volume trend, peak-to-trough decline |

#### Step 3: GARCH(1,1) Volatility (`garch_model.py`)

Fits a GARCH(1,1) model:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

For Nifty 50, typical parameters are α ≈ 0.06, β ≈ 0.92 (α + β ≈ 0.98), confirming strong volatility clustering. Returns are scaled by 100 for numerical conditioning of the maximum likelihood estimator.

#### Step 4: HMM Regime Discovery (`hmm_model.py`)

A 3-state Gaussian HMM with full covariance fits on standardized features:

$$P(x_t \mid z_t = k) \sim \mathcal{N}(\mu_k, \Sigma_k)$$

States are sorted by ascending mean GARCH volatility, guaranteeing that `0 = Low Vol`, `1 = Mid Vol`, `2 = High Vol` regardless of the random initialization.

#### Step 5: XGBoost Prediction (`scripts/regime_classifier.py`)

The critical innovation: HMM labels are **retrospective** (what state is the market in _today_). XGBoost converts them into **prospective** signals (what state will the market be in _tomorrow_):

- **Input:** 15 features on day T
- **Target:** HMM regime label on day T+1
- **Validation:** Walk-forward `TimeSeriesSplit(n_splits=5, gap=5)` — the 5-day gap prevents autoregressive leakage from rolling features and GARCH memory
- **Ensemble:** Averages predictions across all 5 fold models for smoother probability estimates
- **Explainability:** `shap.TreeExplainer` computes exact Shapley values in O(TLD²) time

#### Step 6: MLflow Tracking (`scripts/mlflow_sweep.py`)

Three hyperparameter sweeps logged in `mlflow.db`:

| Run | `n_estimators` | `max_depth` | `learning_rate` | OOF Macro-AUC |
|---|---|---|---|---|
| `sweep_n_est_200` | 200 | 4 | 0.05 | 0.7540 |
| `sweep_n_est_300` | 300 | 4 | 0.05 | **0.7540** |
| `sweep_n_est_500` | 500 | 4 | 0.05 | 0.7540 |

Verify with:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

#### Step 7: Backtest (`backtest.py`)

vectorbt regime-switching strategy with realistic execution assumptions:

| Parameter | Value |
|---|---|
| Initial Capital | ₹1,00,000 |
| Low Vol (0) | 1.0× Long |
| Mid Vol (1) | 0.5× Long |
| High Vol (2) | 0.0× Cash |
| Execution lag | 1 day |
| Fees | 0.1% per trade |
| Slippage | 0.1% |

### Key Design Decisions

| Decision | Implementation | Why |
|---|---|---|
| Walk-forward CV with gap | `TimeSeriesSplit(n_splits=5, gap=5)` | Standard k-fold leaks future data; the 5-day gap prevents GARCH memory contamination |
| Ensemble averaging | Mean of 5 fold model probabilities | Reduces out-of-fold variance without additional training |
| 1-day execution lag | Signal at day T close → execute at T+1 open | Eliminates forward-looking bias in backtest |
| TreeExplainer | `shap.TreeExplainer` (not `KernelExplainer`) | Exact Shapley values in polynomial time vs exponential approximation |
| LRU caching in API | `@lru_cache(maxsize=1)` on data/model loaders | Zero-allocation millisecond responses; data loads once at first request |
| CSV/PKL data contracts | No external database | Each pipeline step reads/writes flat files; independently executable and debuggable |

---

## Evaluation & Benchmarks

### XGBoost Predictive Performance (Walk-Forward CV)

| Metric | Value | Target | Status |
|---|---|---|:---:|
| **OOF Macro-AUC** | **0.7453** | > 0.70 | ✅ PASSED |
| Fold 1 AUC | 0.5000 | — | Monitored |
| Fold 2 AUC | 0.9094 | — | Strong |
| Fold 3 AUC | 0.9844 | — | Excellent |
| Fold 4 AUC | 0.5000 | — | Monitored |
| Fold 5 AUC | 0.8584 | — | Strong |

> **Note on Fold 1 and 4:** AUC of 0.50 indicates the model has no discriminative power in these early/mid folds — expected behavior when training data is insufficient or the market regime distribution is homogeneous. The ensemble average still exceeds the 0.70 threshold.

### Backtest Performance (2014 – Present, Nifty 50)

| Metric | Regime Strategy | Buy & Hold | Difference |
|---|---|---|---|
| **Sharpe Ratio** (Rf = 6.5%) | **0.968** | 0.873 | **+0.095** |
| **Max Drawdown** | **-25.92%** | -38.44% | **+12.52% protected** |
| **Total Return** | +262.01% | +249.48% | +12.53% |
| **CAGR** | 16.61% | 16.13% | +0.48% |
| **Calmar Ratio** | **0.641** | 0.419 | **+0.222** |

Key insight: The regime strategy delivers **comparable returns** with significantly **less drawdown**. The Calmar ratio (return / max drawdown) improvement of +0.222 quantifies the capital protection benefit.

### Testing

24/24 tests passing across 6 modules:

| Module | Tests | Coverage |
|---|---|---|
| `test_api.py` | 6 | All 7 API endpoints, schema validation, error handling |
| `test_models.py` | 5 | XGBoost bundle integrity, AUC thresholds, probability sum |
| `test_features.py` | 4 | Feature count, NaN handling, column names, data types |
| `test_garch.py` | 3 | Stationarity (α+β < 1), volatility bounds, output shape |
| `test_hmm.py` | 3 | State count, label monotonicity, regime name mapping |
| `test_backtest.py` | 3 | Strategy execution, Sharpe calculation, drawdown bounds |

```bash
pytest tests/ -v
```

---

## Where — Project Structure

```
volatility-intelligence-platform/
│
├── data.py                      # Yahoo Finance data ingestion with 7-day cache
├── features.py                  # 15 engineered volatility/momentum/risk features
├── garch_model.py               # GARCH(1,1) conditional volatility estimation
├── hmm_model.py                 # 3-state Gaussian HMM regime discovery + SHAP
├── backtest.py                  # vectorbt regime-switching backtest (1-day lag)
├── run.py                       # Pipeline coordinator (data → features → GARCH → HMM → backtest)
│
├── scripts/
│   ├── regime_classifier.py     # XGBoost walk-forward CV + TreeExplainer SHAP
│   ├── mlflow_tracking.py       # Single-run MLflow logger
│   └── mlflow_sweep.py          # Automated 3-run hyperparameter sweep
│
├── app/
│   ├── main.py                  # FastAPI server (7 endpoints including /predict, /market-summary)
│   └── dashboard.py             # Streamlit prototype dashboard
│
├── frontend/                    # Production Next.js web application
│   ├── pages/index.tsx          # Real-time intelligence dashboard (KPIs, charts, predictions)
│   ├── styles/globals.css       # Tailwind CSS dark theme
│   └── package.json
│
├── models/
│   └── xgb_bundle.pkl           # 5-fold XGBoost walk-forward ensemble + OOF AUC
│
├── results/
│   └── metrics.json             # Platform statistics (Sharpe, AUC, max DD)
│
├── data/                        # Generated data artifacts
│   ├── labeled_data.csv         # Full time series with regime labels + features
│   └── backtest_summary.csv     # Strategy vs B&H performance metrics
│
├── outputs/                     # Generated diagnostic charts
│   ├── garch_vol.png            # GARCH conditional volatility over time
│   ├── regimes.png              # Regime classification timeline
│   ├── shap_summary_xgb.png    # XGBoost SHAP feature importance
│   └── backtest.png             # Strategy equity curve vs benchmark
│
├── research/                    # Quantitative research notebooks
│   ├── 01_arima_null_result.ipynb    # ARIMA baseline (null result — documented for reproducibility)
│   └── 02_sector_momentum_preview.ipynb  # Sector rotation exploration
│
├── tests/                       # Pytest suite (24 tests across 6 modules)
│   ├── test_api.py
│   ├── test_models.py
│   ├── test_features.py
│   ├── test_garch.py
│   ├── test_hmm.py
│   └── test_backtest.py
│
├── mlflow.db                    # SQLite backend for MLflow experiment tracking
├── mlruns/                      # MLflow run artifacts (3 sweeps)
├── Dockerfile                   # Python 3.11 container — trains models during build
├── docker-compose.yml           # Local multi-container orchestration
├── render.yaml                  # Render Blueprint deployment manifest (Docker)
├── vercel.json                  # Vercel frontend deployment manifest
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── ARCHITECTURE.md              # In-depth mathematical and system documentation
└── README.md                    # This file
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- Docker Desktop (optional, for container deployment)

### Option A: Direct Python

```bash
git clone https://github.com/RaajitSingh1306/volatility-intelligence-platform.git
cd volatility-intelligence-platform

pip install -r requirements.txt

# Run base pipeline (data → features → GARCH → HMM → backtest)
python run.py

# Train XGBoost prediction layer
python scripts/regime_classifier.py

# (Optional) Run MLflow hyperparameter sweep
python scripts/mlflow_sweep.py

# Run tests (24/24 passing)
pytest tests/ -v

# Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Option B: Docker

```bash
docker compose build    # Trains models during Docker build
docker compose up
```

```bash
# Verify
curl http://localhost:8000/
# → {"status": "ok", "message": "Volatility Intelligence Platform API v2.0"}

curl http://localhost:8000/predict
# → {"predicted_regime": 0, "predicted_regime_name": "Low Vol", "prob_low": 0.72, ...}
```

### Option C: Next.js Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 for the production dashboard.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — `{"status": "ok", "message": "Volatility Intelligence Platform API v2.0"}` |
| `GET` | `/current` | Latest regime, date, close price, GARCH volatility |
| `GET` | `/predict` | Next-day regime prediction with class probabilities and model AUC |
| `GET` | `/market-summary` | Hero KPIs: regime, GARCH vol, close, Sharpe, max DD, AUC |
| `GET` | `/regimes?last_n=252` | Historical regime time series (default: 252 trading days) |
| `GET` | `/stats` | Per-regime aggregate statistics (days, mean vol, mean return) |
| `GET` | `/backtest-summary` | Strategy vs Buy & Hold performance comparison |

### GET /predict — Response

```json
{
  "predicted_regime": 0,
  "predicted_regime_name": "Low Vol",
  "prob_low": 0.7234,
  "prob_mid": 0.1891,
  "prob_high": 0.0875,
  "model_auc": 0.7453
}
```

### GET /market-summary — Response

```json
{
  "current_regime": "Low Vol",
  "garch_vol_current": 12.45,
  "nifty_close": 24350.75,
  "backtest_sharpe": 0.968,
  "backtest_max_dd": -0.259,
  "model_auc": 0.7453,
  "data_end_date": "2026-09-16"
}
```

---

## Deployment

### Backend → Render

**Option A: Blueprint (1-click)**
1. Push to GitHub
2. Render Dashboard → **New +** → **Blueprint** → select repo
3. Render reads `render.yaml`, configures Docker, sets health check to `/`
4. Click **Apply**

**Option B: Manual**
1. Render → **New +** → **Web Service** → select repo
2. Configure:
   - **Environment**: Docker
   - **Health Check**: `/`
   - **Plan**: Free (note: Docker build trains models and may need >512 MB — consider Starter plan)
3. If using Python runtime instead of Docker:
   - **Build**: `pip install -r requirements.txt && python run.py && python scripts/regime_classifier.py`
   - **Start**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Frontend → Vercel

1. Vercel → **Add New Project** → import repo
2. Set **Root Directory** to `frontend`
3. Add env var: `NEXT_PUBLIC_API_URL` = `https://volatility-intelligence-platform.onrender.com`
4. Deploy

> **Note:** The Docker build trains GARCH, HMM, and XGBoost models during `docker build`. On Render free tier (512 MB RAM), this may OOM with heavy dependencies (mlflow, vectorbt, shap, xgboost). If builds fail, consider the Starter plan or pre-committing trained model artifacts.

---

## Connected Projects

| Project | Role | Repository |
|---|---|---|
| **Volatility Intelligence Platform** (this repo) | Full quant pipeline with prediction + tracking | [volatility-intelligence-platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform) |
| **Volatility Classifier (Simplified)** | Lightweight HMM-only classification | [volatility-classifier-simplified](https://github.com/RaajitSingh1306/volatility-classifier-simplified) |
| **SEBI RAG Bot** | Compliance Q&A assistant — consumes this API via `quant_agent` | [sebi-rag-bot](https://github.com/RaajitSingh1306/sebi-rag-bot) |

The SEBI RAG Bot's `quant_agent` calls this platform's `/current` and `/stats` endpoints to answer market volatility questions within the compliance chat interface.

---

## License & Disclaimer

MIT License. This project is for **educational and research purposes only** and does not constitute investment advice. Past performance of the backtested strategy does not guarantee future results. All backtest results include realistic fees (0.1%) and slippage (0.1%) with a 1-day execution lag.
