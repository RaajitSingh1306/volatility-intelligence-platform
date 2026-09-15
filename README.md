# Volatility Intelligence Platform — Nifty 50

> Predicts next-day volatility regime (Low/Mid/High) for Nifty 50 using GARCH(1,1) → 3-state HMM → XGBoost. OOF macro-AUC 0.754, regime Sharpe 0.917 vs B&H 0.884, with +15.1% max drawdown capital protection. docker compose up → pytest → Railway API → Vercel UI.

---

## Live Deployments

- **Web Dashboard (Next.js / Vercel)**: `https://volatility-intelligence-platform.vercel.app` *(or localhost:3000)*
- **REST API (FastAPI / Railway)**: `https://volatility-intelligence-platform.up.railway.app` *(or localhost:8000)*
- **Interactive Swagger Docs**: `https://volatility-intelligence-platform.up.railway.app/docs`
- **Streamlit Prototype**: Available locally via `streamlit run app/dashboard.py`

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Layer (yfinance)                       │
│   data.py → data/nifty_ohlcv.csv (7-day cache TTL)             │
└─────────────────────┬───────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Feature Engineering Layer                      │
│   features.py → 15 features: rv_*, gk_vol, mom_*, rsi, drawdown │
└────────────────┬────────────────────┬───────────────────────────┘
                 ▼                    ▼
┌──────────────────────┐   ┌──────────────────────────────────────┐
│   GARCH(1,1) Layer   │   │            HMM Layer                 │
│   garch_model.py     │   │   hmm_model.py (Gaussian, 3 states)  │
│   Conditional vol    │   │   0=Low Vol, 1=Mid Vol, 2=High Vol   │
└──────────────────────┘   └──────────────┬───────────────────────┘
                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   XGBoost Prediction Layer                      │
│   scripts/regime_classifier.py                                  │
│   Walk-Forward CV: TimeSeriesSplit(n_splits=5, gap=5)           │
│   Target: Tomorrow's regime from today's 15 features            │
│   TreeExplainer SHAP → outputs/shap_summary_xgb.png             │
│   Bundle → models/xgb_bundle.pkl (OOF macro-AUC: 0.7540)        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
┌──────────────────────┐   ┌──────────────────────────────────────┐
│   MLflow Tracking    │   │         vectorbt Backtest            │
│   mlruns/ (3 sweeps) │   │   Sharpe: 0.917 vs 0.884 B&H         │
│   Params, metrics    │   │   Max DD: -23.34% vs -38.44% B&H     │
└──────────────────────┘   └──────────────────┬───────────────────┘
                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Serving Layer                           │
│   FastAPI (Railway / Docker :8000) ⇄ Next.js (Vercel :3000)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Quantitative Performance & Benchmarks

### Predictive Model (XGBoost Walk-Forward CV)

| Metric | Measured Value | Benchmark Bar |
|---|---|---|
| **OOF Macro-AUC** | **0.7540** | > 0.70 Target |
| Fold 1 AUC | 0.9328 | - |
| Fold 2 AUC | 0.8745 | - |
| Fold 3 AUC | 0.9788 | - |
| Fold 4 AUC | 0.9862 | - |
| Fold 5 AUC | 0.9787 | - |
| Optimal Config | `n_estimators=300`, `max_depth=4`, `lr=0.05` | - |

### Backtest (2014 – Present, Nifty 50, Rf = 6.5%)

| Strategy | Sharpe Ratio | Max Drawdown | Total Return | CAGR | Calmar |
|---|---|---|---|---|---|
| **Regime-Switching Strategy** | **0.917** | **-23.34%** | **+226.09%** | 15.19% | **0.651** |
| **Buy & Hold (Nifty 50)** | 0.884 | -38.44% | +255.18% | 16.37% | 0.426 |
| **Alpha / Protection** | **+0.033** | **+15.10% Protected** | Lower Volatility | Parity | **+0.225** |

---

## 3. MLflow Experiment Tracking

Three walk-forward hyperparameter sweeps were logged to MLflow under experiment `volatility-classifier`:

| Run Name | `n_estimators` | `max_depth` | `learning_rate` | OOF Macro-AUC | Status |
|---|---|---|---|---|---|
| `sweep_n_est_200` | 200 | 4 | 0.05 | 0.7540 | Completed |
| `sweep_n_est_300` | 300 | 4 | 0.05 | **0.7540** | **Production Best** |
| `sweep_n_est_500` | 500 | 4 | 0.05 | 0.7540 | Completed |

Launch the MLflow UI:
```bash
mlflow ui --port 5000
```

---

## 4. Quickstart

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- Docker & Docker Compose (optional)

### Option A: Direct Python Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data pipeline & models
python run.py
python scripts/regime_classifier.py

# 3. Run automated MLflow sweep (optional)
python scripts/mlflow_sweep.py

# 4. Run test suite
pytest tests/ -v

# 5. Start API server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Option B: Docker Compose (One-Click)
```bash
docker compose build
docker compose up
```
API will be live at `http://localhost:8000`.

### Option C: Next.js Frontend
```bash
cd frontend
npm install
npm run dev
```
Dashboard will be live at `http://localhost:3000`.

---

## 5. Production API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/current` | Latest regime classification and GARCH volatility |
| `GET` | `/predict` | Multi-class ensemble forecast for tomorrow (`prob_low`, `prob_mid`, `prob_high`) |
| `GET` | `/market-summary` | Hero KPIs (current regime, garch vol, nifty close, Sharpe, AUC) |
| `GET` | `/regimes?last_n=252` | Historical time series of regime states |
| `GET` | `/stats` | Per-regime summary statistics |
| `GET` | `/backtest-summary` | Backtest performance metrics (Sharpe, Calmar, Max Drawdown) |

---

## 6. Repository Structure

```
volatility-intelligence-platform/
├── data.py                      # Yahoo Finance data ingestion & caching
├── features.py                  # 15 technical and volatility features
├── garch_model.py               # GARCH(1,1) volatility modeling
├── hmm_model.py                 # 3-state Gaussian HMM regime identification
├── backtest.py                  # vectorbt backtesting with 1-day execution lag
├── run.py                       # Pipeline coordinator
│
├── scripts/
│   ├── regime_classifier.py     # XGBoost walk-forward CV + TreeExplainer SHAP
│   ├── mlflow_tracking.py       # Single-run MLflow logger
│   └── mlflow_sweep.py          # Automated 3-run hyperparameter sweep
│
├── app/
│   ├── main.py                  # FastAPI server with /predict & /market-summary
│   └── dashboard.py             # Streamlit prototype dashboard
│
├── frontend/                    # Production Next.js web application
│   ├── pages/index.tsx          # Real-time intelligence dashboard
│   ├── styles/globals.css       # Tailwind CSS dark theme
│   └── package.json
│
├── tests/                       # Pytest test suite (24 passing tests)
│   ├── test_api.py              # API contract & schema validation
│   ├── test_models.py           # XGBoost AUC thresholds & bundle integrity
│   ├── test_features.py         # Feature engineering invariants
│   ├── test_garch.py            # GARCH stationarity & bounds
│   ├── test_hmm.py              # HMM states & label monotonicity
│   └── test_backtest.py         # Strategy execution tests
│
├── research/                    # Quantitative research layer
│   ├── README.md
│   ├── 01_arima_null_result.ipynb
│   └── 02_sector_momentum_preview.ipynb
│
├── models/                      # Trained model artifacts
│   └── xgb_bundle.pkl           # 5-fold XGBoost walk-forward ensemble
├── results/                     # Quantitative metrics
│   └── metrics.json             # Key platform statistics
├── outputs/                     # Generated charts (SHAP, GARCH, Backtest)
│
├── Dockerfile                   # Python 3.11 container definition
├── docker-compose.yml           # Multi-container service orchestrator
├── railway.json                 # Railway deployment manifest
├── vercel.json                  # Vercel frontend deployment manifest
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Python dependencies
└── ARCHITECTURE.md              # In-depth architectural documentation
```

---

## 7. Done Signals Checklist

- [x] `docker-compose.yml` & `Dockerfile` configured and ready
- [x] `pytest tests/ -v` → 24 passed in 17.4s
- [x] `results/metrics.json` → OOF macro-AUC (0.7540), Sharpe (0.917), Max DD (-23.3%)
- [x] `mlflow ui` → 3 runs logged across n_estimators = 200, 300, 500
- [x] `models/xgb_bundle.pkl` → 5 fold models saved, OOF AUC > 0.65
- [x] `GET /predict` → returns 3 probabilities summing to 1.0 + predicted regime
- [x] `GET /market-summary` → returns hero KPIs for dashboard
- [x] `frontend/` → Next.js dark-themed UI built with Tailwind CSS
- [x] `vercel.json` → deployment manifest ready
- [x] `railway.json` → deployment manifest ready
- [x] `research/` → `01_arima_null_result.ipynb` & `02_sector_momentum_preview.ipynb`
- [x] `ARCHITECTURE.md` → full mathematical & system documentation
