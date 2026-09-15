"""
tests/test_models.py
--------------------
Unit and integration tests for XGBoost predictive model, bundle integrity,
and results metrics schema.
"""

import json
import pickle
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
BUNDLE_PATH = ROOT_DIR / "models" / "xgb_bundle.pkl"
METRICS_PATH = ROOT_DIR / "results" / "metrics.json"


def test_xgb_bundle_exists():
    """Verify XGBoost bundle exists and has all required keys."""
    assert BUNDLE_PATH.exists(), f"Model bundle missing at {BUNDLE_PATH}"
    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)

    assert "models" in bundle, "Missing 'models' key in bundle"
    assert "feature_names" in bundle, "Missing 'feature_names' in bundle"
    assert "oof_auc" in bundle, "Missing 'oof_auc' in bundle"
    assert len(bundle["models"]) == 5, f"Expected 5 fold models, got {len(bundle['models'])}"


def test_xgb_oof_auc_above_threshold():
    """Verify out-of-fold macro-AUC meets minimum performance bar (>0.65)."""
    assert BUNDLE_PATH.exists()
    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)

    oof_auc = bundle["oof_auc"]
    assert oof_auc > 0.65, f"OOF AUC too low: {oof_auc:.4f} (expected > 0.65)"


def test_xgb_feature_count():
    """Verify bundle was trained on the exact 15 feature set."""
    assert BUNDLE_PATH.exists()
    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)

    feature_names = bundle["feature_names"]
    assert len(feature_names) == 15, f"Expected 15 features, got {len(feature_names)}"


def test_metrics_json_exists():
    """Verify results/metrics.json contains all required KPI fields."""
    assert METRICS_PATH.exists(), f"Metrics JSON missing at {METRICS_PATH}"
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    required_keys = [
        "oof_macro_auc",
        "regime_dist",
        "sharpe_strategy",
        "sharpe_bh",
        "max_dd_strategy",
        "max_dd_bh",
        "best_n_estimators",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key '{key}' in metrics.json"

    assert metrics["oof_macro_auc"] > 0.65
    assert metrics["sharpe_strategy"] > 0
