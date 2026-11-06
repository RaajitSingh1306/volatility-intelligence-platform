"""
mlflow_sweep.py
---------------
Automated MLflow hyperparameter sweep across n_estimators = [200, 300, 500].
Logs all runs to MLflow, prints a comparison table, and persists the best model
bundle as the production models/xgb_bundle.pkl and updates results/metrics.json.

Usage:
    python scripts/mlflow_sweep.py
"""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
from scripts.mlflow_tracking import log_regime_run
from scripts.regime_classifier import train_regime_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mlflow_sweep")

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"


def run_sweep() -> List[Dict[str, Any]]:
    """
    Run 3-experiment sweep (n_estimators = 200, 300, 500).
    """
    sweep_configs = [200, 300, 500]
    run_results = []

    logger.info("==================================================")
    logger.info("Starting MLflow 3-Run Hyperparameter Sweep")
    logger.info("Configs: n_estimators in %s", sweep_configs)
    logger.info("==================================================")

    for n_est in sweep_configs:
        logger.info("\n>>> Running experiment with n_estimators = %d ...", n_est)
        res = log_regime_run(
            n_estimators=n_est,
            max_depth=4,
            learning_rate=0.05,
            run_name=f"sweep_n_est_{n_est}",
        )
        run_results.append(res)

    # Print comparison table
    print("\n" + "=" * 65)
    print(f"{'Run Name':<20} | {'n_est':<8} | {'OOF macro-AUC':<15} | {'Run ID':<15}")
    print("-" * 65)
    best_res = None
    for r in run_results:
        print(f"{r['run_name']:<20} | {r['n_estimators']:<8} | {r['oof_auc']:<15.4f} | {r['run_id'][:12]}...")
        if best_res is None or r["oof_auc"] > best_res["oof_auc"]:
            best_res = r
    print("=" * 65)

    print(f"\nBest Config: n_estimators = {best_res['n_estimators']} (OOF macro-AUC: {best_res['oof_auc']:.4f})")

    # Retrain/save with best config to ensure production models/xgb_bundle.pkl is optimal
    logger.info("Saving best model (n_estimators=%d) to production bundle...", best_res["n_estimators"])
    final_res = train_regime_classifier(
        n_estimators=best_res["n_estimators"],
        max_depth=best_res["max_depth"],
        learning_rate=best_res["learning_rate"],
        save_bundle=True,
        bundle_name="xgb_bundle.pkl",
    )

    # Update metrics.json with best_n_estimators
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            m = json.load(f)
        m["best_n_estimators"] = best_res["n_estimators"]
        m["oof_macro_auc"] = round(best_res["oof_auc"], 4)
        with open(metrics_path, "w") as f:
            json.dump(m, f, indent=2)

    print(f"Production bundle saved -> models/xgb_bundle.pkl")
    print(f"Metrics updated -> results/metrics.json")
    print("Review in MLflow UI: mlflow ui --port 5000\n")

    return run_results


if __name__ == "__main__":
    run_sweep()
