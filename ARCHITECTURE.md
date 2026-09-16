# Volatility Intelligence Platform — System Architecture

Comprehensive architectural blueprint, mathematical foundations, model chaining, and design justifications for the Nifty 50 Volatility Intelligence Platform.

---

## 1. End-to-End System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Data Layer                             │
│   Yahoo Finance (yfinance) → data.py → data/nifty_ohlcv.csv     │
│   Caching: 7-day TTL. Re-download if cache stale.               │
└─────────────────────┬───────────────────────────────────────────┘
                      │ OHLCV: Date, Open, High, Low, Close, Volume
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Feature Engineering Layer                      │
│   features.py → 15 features:                                    │
│     - Realized Vol:    rv_5d, rv_10d, rv_21d, rv_63d            │
│     - Vol-of-Vol:      vol_ratio_5_21, vol_ratio_21_63          │
│     - OHLC Estimators: gk_vol_21 (Garman-Klass), park_vol_21    │
│     - Momentum:        mom_5d, mom_10d, mom_21d, mom_63d        │
│     - Microstructure:  rsi_14, vol_ma_ratio, drawdown           │
│   → data/features.csv                                           │
└────────────────┬────────────────────┬───────────────────────────┘
                 │                    │
                 ▼                    ▼
┌──────────────────────┐   ┌──────────────────────────────────────┐
│   GARCH(1,1) Layer   │   │            HMM Layer                 │
│   garch_model.py     │   │   hmm_model.py                       │
│   Conditional vol    │   │   Gaussian HMM, 3 states             │
│   → garch_vol series │   │   Full covariance matrix             │
│   → outputs/         │   │   Sorted by mean conditional vol     │
│     garch_vol.png    │   │   → regime labels: 0=Low, 1=Mid, 2=High
└──────────────────────┘   │   → data/labeled_data.csv            │
                           └──────────────┬───────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   XGBoost Prediction Layer                      │
│   scripts/regime_classifier.py                                  │
│   Input:   X = 15 features on day T                             │
│   Target:  y = HMM regime label on day T+1 (tomorrow's state)   │
│   CV:      TimeSeriesSplit(n_splits=5, gap=5)                   │
│   Model:   XGBClassifier (multi:softprob, 3 classes)            │
│   Explain: TreeExplainer SHAP exact attributions                │
│   Artifacts:                                                    │
│     → models/xgb_bundle.pkl (5-model walk-forward ensemble)     │
│     → outputs/shap_summary_xgb.png                              │
│     → results/metrics.json                                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
┌──────────────────────┐   ┌──────────────────────────────────────┐
│   MLflow Tracking    │   │         vectorbt Backtest            │
│   scripts/mlflow_    │   │   backtest.py                        │
│   tracking.py &      │   │   Position rule:                     │
│   mlflow_sweep.py    │   │     Regime 0 (Low)  → 1.0x Long      │
│   Logs: params, fold │   │     Regime 1 (Mid)  → 0.5x Long      │
│   AUCs, OOF AUC,     │   │     Regime 2 (High) → 0.0x Cash      │
│   artifacts to       │   │   Execution: 1-day lag, 0.10% fees   │
│   mlruns/            │   │   → data/backtest_summary.csv        │
└──────────────────────┘   │   → outputs/backtest.png             │
                           └──────────────────┬───────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Serving Layer                           │
│                                                                 │
│   FastAPI Engine (Render / Docker)   Next.js UI (Vercel)        │
│   app/main.py                        frontend/pages/index.tsx   │
│   - GET /current                     - Real-time KPI cards      │
│   - GET /regimes?last_n=252          - Predicted Regime Prob    │
│   - GET /stats                       - SHAP feature breakdown   │
│   - GET /backtest-summary            - Strategy equity curve    │
│   - GET /predict                                                │
│   - GET /market-summary                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Chain & Mathematical Justifications

### 2.1 GARCH(1,1) Conditional Volatility
Daily returns $r_t$ follow near-martingale behavior (EMH), but squared returns $r_t^2$ exhibit pronounced persistence (ARCH effect).
The conditional variance $\sigma_t^2$ is modeled as:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

- $\omega$: Long-run baseline variance target.
- $\alpha$: Impact of yesterday's shock (innovation).
- $\beta$: Persistence of volatility memory.
- Stationarity condition: $\alpha + \beta < 1$. For Nifty 50, $\alpha \approx 0.06$, $\beta \approx 0.92$, yielding $\alpha + \beta \approx 0.98$, confirming long-memory volatility clustering.
- **Numerical conditioning**: Returns are scaled by 100 before estimation to provide a well-conditioned optimization landscape for maximum likelihood estimation.

### 2.2 Hidden Markov Model (HMM) Regime Detection
Rather than relying on arbitrary volatility thresholds, a 3-state Gaussian HMM models latent market states:

$$\theta = (\pi, A, B)$$

- $\pi$: Initial state distribution.
- $A$: Transition probability matrix ($A_{i,j} = P(z_t = j \mid z_{t-1} = i)$).
- $B$: Emission distributions $P(x_t \mid z_t = k) \sim \mathcal{N}(\mu_k, \Sigma_k)$ with full covariance across all 15 features.
- **State sorting**: Hidden state indices are inherently arbitrary; our pipeline maps states by ascending mean `garch_vol` such that `0 = Low Vol`, `1 = Mid Vol`, `2 = High Vol`, guaranteeing invariant semantics across runs.

### 2.3 XGBoost Predictive Layer & Walk-Forward Validation
The HMM retrospectively assigns regime labels to past days. To convert retrospective classification into forward trading signals, an XGBoost gradient boosting ensemble is trained to predict $z_{t+1}$ given features $x_t$:

$$P(z_{t+1} = k \mid x_t) = \frac{e^{F_k(x_t)}}{\sum_{j=0}^2 e^{F_j(x_t)}}$$

- **TimeSeriesSplit with `gap=5`**: Standard $K$-fold cross-validation introduces lookahead leakage. Walk-forward expansion keeps historical temporal integrity. The 5-day gap eliminates autoregressive leakage from rolling features and GARCH volatility memory.
- **TreeExplainer SHAP**: Unlike model-agnostic `KernelExplainer` which requires $O(2^N)$ sampling approximations, `TreeExplainer` computes exact Shapley attributions in $O(TLD^2)$ time directly through tree traversal.

---

## 3. Key Design Decisions

| Architectural Choice | Implementation | Rationale |
|---|---|---|
| **Leakage Buffer** | `TimeSeriesSplit(n_splits=5, gap=5)` | Prevents GARCH lag leakage across training and validation folds. |
| **Ensemble Inference** | Average probabilities across 5 fold models | Smooths out-of-fold variance and avoids overconfidence. |
| **Execution Lag** | 1-day lag on backtest signals | Signal generated at day $T$ close executes at day $T+1$ open. |
| **SHAP Explainer** | `shap.TreeExplainer` | Exact, deterministic tree-based Shapley feature attributions. |
| **Caching Tier** | `@lru_cache(maxsize=1)` on models/data | Zero-allocation millisecond response times in FastAPI. |
| **State Decoupling** | CSV/PKL data contracts | Independent component execution without external database overhead. |

---

## 4. Platform Performance Benchmarks

### Predictive Performance (XGBoost Walk-Forward CV)

| Metric | Measured Value | Minimum Bar | Status |
|---|---|---|---|
| **OOF Macro-AUC** | **0.7540** | > 0.70 | PASS |
| Fold 1 AUC | 0.9328 | - | PASS |
| Fold 2 AUC | 0.8745 | - | PASS |
| Fold 3 AUC | 0.9788 | - | PASS |
| Fold 4 AUC | 0.9862 | - | PASS |
| Fold 5 AUC | 0.9787 | - | PASS |
| Best Hyperparameter | $n\_estimators = 300$ | - | Optimal |

### Backtest Performance (2014 – Present, Nifty 50)

| Metric | Regime Strategy | Buy & Hold Benchmark | Alpha / Difference |
|---|---|---|---|
| **Sharpe Ratio (Rf=6.5%)** | **0.917** | 0.884 | **+0.033** |
| **Max Drawdown** | **-23.34%** | **-38.44%** | **+15.10% Capital Protection** |
| Calmar Ratio | 0.651 | 0.426 | +0.225 |
| Total Return | +226.09% | +255.18% | Lower beta risk profile |
| Annualized CAGR | 15.19% | 16.37% | Risk-adjusted parity |
| Win Rate | 100.0% | 100.0% | Multi-year holding |

---

## 5. API Specification

| Route | Method | Purpose | Response Schema |
|---|---|---|---|
| `/` | `GET` | Health status | `{"status": "ok", "message": "..."}` |
| `/current` | `GET` | Latest regime & volatility | `CurrentRegime(date, regime, regime_name, garch_vol_pct, close)` |
| `/regimes` | `GET` | Historical timeseries (`last_n`) | `list[RegimePoint(date, close, regime, regime_name, ...)]` |
| `/stats` | `GET` | Aggregate regime statistics | `list[dict(regime_name, days, mean_vol_pct, mean_ret_pct)]` |
| `/backtest-summary` | `GET` | Strategy vs B&H metrics | `dict(Total Return, CAGR, Sharpe Ratio, Max Drawdown, ...)` |
| `/predict` | `GET` | Next-day probability prediction | `PredictResponse(predicted_regime, prob_low, prob_mid, prob_high, model_auc)` |
| `/market-summary` | `GET` | Hero dashboard overview | `MarketSummaryResponse(current_regime, garch_vol_current, nifty_close, sharpe, ...)` |
