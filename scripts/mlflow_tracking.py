"""
mlflow_tracking.py
------------------
Single-run MLflow experiment tracking wrapper for XGBoost Volatility Classifier.
Logs hyperparameters, fold AUCs, OOF macro-AUC, and model artifacts to MLflow.

Usage:
    python scripts/mlflow_tracking.py
    python scripts/mlflow_tracking.py --n_estimators 300 --max_depth 4
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
from scripts.regime_classifier import train_regime_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mlflow_tracking")

EXPERIMENT_NAME = "volatility-classifier"


def log_regime_run(
    n_estimators: int = 500,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50,
    experiment_name: str = EXPERIMENT_NAME,
    run_name: str = None,
) -> Dict[str, Any]:
    """
    Train and log a single run of the XGBoost regime classifier to MLflow.
    """
    mlflow.set_experiment(experiment_name)
    run_name = run_name or f"xgb_n_est_{n_estimators}_depth_{max_depth}"

    with mlflow.start_run(run_name=run_name) as run:
        logger.info("Started MLflow run: %s (ID: %s)", run_name, run.info.run_id)

        # Log parameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "early_stopping_rounds": early_stopping_rounds,
            "n_splits": 5,
            "gap": 5,
            "objective": "multi:softprob",
            "num_class": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        })

        # Train model
        res = train_regime_classifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            save_bundle=True,
        )

        oof_auc = res["oof_auc"]
        fold_aucs = res["fold_aucs"]

        # Log metrics
        mlflow.log_metric("oof_macro_auc", oof_auc)
        for fold_idx, f_auc in enumerate(fold_aucs, 1):
            mlflow.log_metric(f"fold_{fold_idx}_auc", f_auc)

        # Log artifacts
        bundle_path = Path(res["bundle_path"])
        shap_path = Path(res["shap_path"])
        metrics_path = Path(res["metrics_path"])

        if bundle_path.exists():
            mlflow.log_artifact(str(bundle_path), artifact_path="models")
        if shap_path.exists():
            mlflow.log_artifact(str(shap_path), artifact_path="figures")
        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        logger.info("MLflow run %s completed with OOF macro-AUC: %.4f", run.info.run_id, oof_auc)

        return {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "oof_auc": oof_auc,
            "fold_aucs": fold_aucs,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track XGBoost training in MLflow")
    parser.add_argument("--n_estimators", type=int, default=500, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=4, help="Max depth")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--early_stopping_rounds", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--experiment_name", type=str, default=EXPERIMENT_NAME, help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name")
    args = parser.parse_args()

    # Support N_EST environment variable as alternative shorthand
    if "N_EST" in os.environ and args.n_estimators == 500:
        args.n_estimators = int(os.environ["N_EST"])

    result = log_regime_run(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    print(f"\nSuccessfully logged MLflow run {result['run_id']} with OOF macro-AUC: {result['oof_auc']:.4f}")
