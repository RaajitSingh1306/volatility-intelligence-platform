import React, { useState, useEffect } from "react";
import Head from "next/head";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  Clock,
  ExternalLink,
  Layers,
  LineChart,
  RefreshCw,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
  Zap,
} from "lucide-react";

interface MarketSummary {
  current_regime: string;
  garch_vol_current: number;
  nifty_close: number;
  backtest_sharpe: number;
  backtest_max_dd: number;
  model_auc: number;
  data_end_date: string;
}

interface PredictData {
  predicted_regime: number;
  predicted_regime_name: string;
  prob_low: number;
  prob_mid: number;
  prob_high: number;
  model_auc: number;
}

interface RegimePoint {
  date: string;
  close: number;
  regime: number;
  regime_name: string;
  garch_vol_pct: number;
  returns_pct: number;
}

const DEFAULT_SUMMARY: MarketSummary = {
  current_regime: "Low Vol",
  garch_vol_current: 9.16,
  nifty_close: 23779.15,
  backtest_sharpe: 0.917,
  backtest_max_dd: -0.233,
  model_auc: 0.754,
  data_end_date: "2026-09-07",
};

const DEFAULT_PREDICT: PredictData = {
  predicted_regime: 0,
  predicted_regime_name: "Low Vol",
  prob_low: 0.9348,
  prob_mid: 0.0651,
  prob_high: 0.0001,
  model_auc: 0.754,
};

const TOP_SHAP_FEATURES = [
  { name: "rv_21d (21-day Realized Vol)", value: 1.236, pct: 100 },
  { name: "gk_vol_21 (Garman-Klass Vol)", value: 0.484, pct: 39 },
  { name: "drawdown (Rolling Drawdown)", value: 0.426, pct: 34 },
  { name: "rv_63d (63-day Realized Vol)", value: 0.409, pct: 33 },
  { name: "park_vol_21 (Parkinson Vol)", value: 0.320, pct: 26 },
  { name: "rv_10d (10-day Realized Vol)", value: 0.294, pct: 24 },
  { name: "mom_63d (3-Month Momentum)", value: 0.293, pct: 24 },
];

export default function Home() {
  const [summary, setSummary] = useState<MarketSummary>(DEFAULT_SUMMARY);
  const [predict, setPredict] = useState<PredictData>(DEFAULT_PREDICT);
  const [history, setHistory] = useState<RegimePoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<string>("");

  const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sumRes, predRes, histRes] = await Promise.all([
        fetch(`${API}/market-summary`).then((r) => {
          if (!r.ok) throw new Error("Failed to load market summary");
          return r.json();
        }),
        fetch(`${API}/predict`).then((r) => {
          if (!r.ok) throw new Error("Failed to load prediction");
          return r.json();
        }),
        fetch(`${API}/regimes?last_n=8`).then((r) => {
          if (!r.ok) return [];
          return r.json();
        }),
      ]);

      setSummary(sumRes);
      setPredict(predRes);
      setHistory(histRes);
      setLastRefreshed(new Date().toLocaleTimeString());
    } catch (err: any) {
      console.warn("API offline or loading error, fallback values active:", err);
      setError("Backend API is currently offline. Showing cached benchmark telemetry.");
      setLastRefreshed(new Date().toLocaleTimeString());
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const getRegimeColor = (name: string) => {
    if (name.includes("Low")) return "text-emerald-400 border-emerald-500/30 bg-emerald-500/10";
    if (name.includes("Mid")) return "text-amber-400 border-amber-500/30 bg-amber-500/10";
    return "text-rose-400 border-rose-500/30 bg-rose-500/10";
  };

  const getRegimeBadge = (name: string) => {
    if (name.includes("Low")) {
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          Low Vol (Risk-On / Trending)
        </span>
      );
    }
    if (name.includes("Mid")) {
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/40">
          <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
          Mid Vol (Transitional)
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/40">
        <span className="w-2 h-2 rounded-full bg-rose-400 animate-pulse"></span>
        High Vol (Risk-Off / Capital Protect)
      </span>
    );
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 selection:bg-blue-600 selection:text-white pb-16">
      <Head>
        <title>Volatility Intelligence Platform | Nifty 50</title>
        <meta
          name="description"
          content="Multi-layer volatility regime prediction using GARCH(1,1), Hidden Markov Models, and XGBoost with walk-forward cross-validation."
        />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      {/* Top Navigation */}
      <header className="sticky top-0 z-50 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                  Volatility Intelligence Platform
                </span>
                <span className="px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded bg-blue-500/10 text-blue-400 border border-blue-500/30">
                  NIFTY 50
                </span>
              </div>
              <p className="text-xs text-slate-400">GARCH(1,1) → 3-State HMM → XGBoost Walk-Forward</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-400">
              <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              API: <span className="text-slate-200 font-mono">{API}</span>
            </div>
            <button
              onClick={fetchData}
              disabled={loading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 active:scale-95 border border-slate-700 text-xs font-medium text-slate-200 transition"
              title="Refresh telemetry"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-blue-400" : ""}`} />
              <span>Refresh</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8 space-y-8">
        {/* Banner Alert if Offline */}
        {error && (
          <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/10 text-amber-300 text-sm flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 flex-shrink-0" />
              <span>{error} Backend is either starting up or unreachable. Displaying baseline validated telemetry.</span>
            </div>
            <span className="text-xs text-amber-400/80 font-mono">Last updated: {lastRefreshed}</span>
          </div>
        )}

        {/* Hero Section Banner */}
        <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900/90 to-slate-950 p-6 sm:p-8">
          <div className="absolute top-0 right-0 -mt-8 -mr-8 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl pointer-events-none"></div>
          <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 text-xs font-semibold rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                  SYSTEM READY
                </span>
                <span className="text-xs text-slate-400">Trading Session Date: {summary.data_end_date}</span>
              </div>
              <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                Predictive Volatility Regime Intelligence
              </h1>
              <p className="text-sm text-slate-400 max-w-2xl leading-relaxed">
                Retrospective HMM classification identifies market states. An XGBoost walk-forward CV ensemble
                predicts next-day regimes, preventing capital drawdowns with a 1-day execution lag.
              </p>
            </div>

            <div className="flex items-center gap-4 flex-wrap">
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800/80 min-w-[140px]">
                <div className="text-xs text-slate-400">Nifty 50 Close</div>
                <div className="text-xl font-bold text-white tracking-tight">
                  ₹{summary.nifty_close.toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                </div>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800/80 min-w-[140px]">
                <div className="text-xs text-slate-400">OOF Macro-AUC</div>
                <div className="text-xl font-bold text-emerald-400 tracking-tight flex items-center gap-1">
                  <span>{(summary.model_auc * 100).toFixed(1)}%</span>
                  <span className="text-[10px] text-slate-400 font-normal">(Target &gt;70%)</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Primary KPIs Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Card 1: Today's State */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 flex flex-col justify-between hover:border-slate-700 transition">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-blue-400" />
                  Today's Regime State
                </span>
                <span className="text-xs text-slate-500">HMM Output</span>
              </div>

              <div>
                <div className="text-2xl font-black tracking-tight text-white mb-2">
                  {summary.current_regime}
                </div>
                {getRegimeBadge(summary.current_regime)}
              </div>

              <div className="pt-2 border-t border-slate-800/60 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-400">GARCH(1,1) Annualized Vol:</span>
                  <span className="font-mono font-bold text-blue-400">{summary.garch_vol_current}%</span>
                </div>
                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-blue-500 h-full rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(summary.garch_vol_current * 3, 100)}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-[10px] text-slate-500">
                  <span>0% (Calm)</span>
                  <span>15% (Normal)</span>
                  <span>30%+ (Stress)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Card 2: Tomorrow's XGBoost Prediction */}
          <div className="rounded-2xl border border-blue-500/40 bg-gradient-to-b from-blue-950/20 to-slate-900/60 p-6 flex flex-col justify-between shadow-lg shadow-blue-500/5 relative overflow-hidden">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-blue-400 flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5" />
                  Tomorrow's Forecast
                </span>
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-blue-500/20 text-blue-300 border border-blue-500/30">
                  XGBoost 5-Fold
                </span>
              </div>

              <div>
                <div className="text-2xl font-black tracking-tight text-white mb-2">
                  {predict.predicted_regime_name}
                </div>
                <div className="text-xs text-slate-400">
                  Model Probability: <span className="text-white font-bold font-mono">{(Math.max(predict.prob_low, predict.prob_mid, predict.prob_high) * 100).toFixed(1)}%</span>
                </div>
              </div>

              <div className="space-y-2.5 pt-2 border-t border-slate-800/60 text-xs">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">P(Low Vol):</span>
                    <span className="font-mono text-emerald-400 font-bold">{(predict.prob_low * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div className="bg-emerald-400 h-full rounded-full transition-all duration-500" style={{ width: `${predict.prob_low * 100}%` }}></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">P(Mid Vol):</span>
                    <span className="font-mono text-amber-400 font-bold">{(predict.prob_mid * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div className="bg-amber-400 h-full rounded-full transition-all duration-500" style={{ width: `${predict.prob_mid * 100}%` }}></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">P(High Vol):</span>
                    <span className="font-mono text-rose-400 font-bold">{(predict.prob_high * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div className="bg-rose-400 h-full rounded-full transition-all duration-500" style={{ width: `${predict.prob_high * 100}%` }}></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Card 3: Backtest & Risk Mitigation */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 flex flex-col justify-between hover:border-slate-700 transition">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <ShieldAlert className="w-3.5 h-3.5 text-emerald-400" />
                  Strategy Alpha & Protection
                </span>
                <span className="text-xs text-slate-500">2014 – 2026</span>
              </div>

              <div>
                <div className="text-2xl font-black tracking-tight text-white mb-1 flex items-center gap-2">
                  <span>Sharpe {summary.backtest_sharpe.toFixed(3)}</span>
                  <span className="text-xs px-2 py-0.5 rounded font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                    vs 0.884 B&amp;H
                  </span>
                </div>
                <p className="text-xs text-slate-400">Risk-free hurdle = 6.50% (India 10Y G-Sec)</p>
              </div>

              <div className="pt-2 border-t border-slate-800/60 space-y-2 text-xs">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Strategy Max Drawdown:</span>
                  <span className="font-mono font-bold text-emerald-400">{(summary.backtest_max_dd * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Buy &amp; Hold Max Drawdown:</span>
                  <span className="font-mono font-bold text-rose-400">-38.4%</span>
                </div>
                <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-[11px] font-medium flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                  <span>+15.1% Capital Protection during market drawdowns</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Deep Dive: SHAP Feature Importance & History */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* SHAP Feature Importance */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-blue-400" />
                  SHAP Feature Attribution (TreeExplainer)
                </h3>
                <p className="text-xs text-slate-400">
                  Exact Shapley values explaining which features drive tomorrow's regime prediction.
                </p>
              </div>
              <span className="text-xs text-slate-500 font-mono">15 Features</span>
            </div>

            <div className="space-y-3 pt-2">
              {TOP_SHAP_FEATURES.map((item, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300 font-medium">{item.name}</span>
                    <span className="text-slate-400 font-mono">{item.value.toFixed(3)}</span>
                  </div>
                  <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-blue-600 to-cyan-400 h-full rounded-full transition-all duration-500"
                      style={{ width: `${item.pct}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Regime Time Series Feed */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <LineChart className="w-4 h-4 text-emerald-400" />
                  Recent Regime Classifications
                </h3>
                <p className="text-xs text-slate-400">
                  Historical sequence of daily closes, regimes, and conditional volatilities.
                </p>
              </div>
              <span className="text-xs text-slate-500 font-mono">Latest 8 Days</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2 px-2">Date</th>
                    <th className="py-2 px-2">Close</th>
                    <th className="py-2 px-2">Regime</th>
                    <th className="py-2 px-2 text-right">GARCH Vol</th>
                    <th className="py-2 px-2 text-right">Return</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {history.length > 0 ? (
                    history.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-800/30 transition">
                        <td className="py-2 px-2 text-slate-300">{row.date}</td>
                        <td className="py-2 px-2 text-slate-200 font-bold">₹{row.close.toFixed(2)}</td>
                        <td className="py-2 px-2">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                              row.regime === 0
                                ? "bg-emerald-500/20 text-emerald-400"
                                : row.regime === 1
                                ? "bg-amber-500/20 text-amber-400"
                                : "bg-rose-500/20 text-rose-400"
                            }`}
                          >
                            {row.regime_name}
                          </span>
                        </td>
                        <td className="py-2 px-2 text-right text-blue-400">{row.garch_vol_pct.toFixed(2)}%</td>
                        <td
                          className={`py-2 px-2 text-right font-bold ${
                            row.returns_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                          }`}
                        >
                          {row.returns_pct >= 0 ? "+" : ""}
                          {row.returns_pct.toFixed(2)}%
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={5} className="py-6 text-center text-slate-500">
                        Loading recent market classifications...
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* System Methodology Footnote */}
        <div className="rounded-xl border border-slate-800/60 bg-slate-900/30 p-5 text-xs text-slate-400 space-y-2">
          <div className="flex items-center gap-2 text-slate-300 font-semibold">
            <Layers className="w-4 h-4 text-blue-400" />
            <span>Methodology &amp; Anti-Leakage Protocol</span>
          </div>
          <p className="leading-relaxed">
            The platform addresses time-series leakage through a strict 5-day gap walk-forward cross-validation
            schedule (<code className="text-blue-300">TimeSeriesSplit(n_splits=5, gap=5)</code>). The 5-day gap prevents
            autoregressive correlation from GARCH(1,1) volatility persistence from leaking into test folds. Backtested
            signals incorporate a 1-day execution lag to simulate real institutional trade settlement.
          </p>
        </div>
      </main>
    </div>
  );
}
