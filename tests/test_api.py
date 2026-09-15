"""
tests/test_api.py
-----------------
API contract tests verifying response schemas, status codes, and data invariants.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """GET / returns 200 with status ok."""
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"


def test_current_endpoint():
    """GET /current returns valid regime name and values."""
    r = client.get("/current")
    assert r.status_code == 200
    data = r.json()
    assert "regime_name" in data
    assert data["regime_name"] in ["Low Vol", "Mid Vol", "High Vol"]
    assert "garch_vol_pct" in data
    assert data["garch_vol_pct"] > 0
    assert "close" in data
    assert data["close"] > 0


def test_regimes_endpoint():
    """GET /regimes returns historical array."""
    r = client.get("/regimes?last_n=20")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 20
    for pt in data:
        assert "date" in pt
        assert "close" in pt
        assert "regime" in pt
        assert pt["regime"] in [0, 1, 2]


def test_stats_endpoint():
    """GET /stats returns regime aggregate metrics."""
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    for stat in data:
        assert "regime_name" in stat
        assert "days" in stat
        assert "mean_vol_pct" in stat


def test_predict_endpoint():
    """GET /predict returns multi-class probabilities summing to ~1.0."""
    r = client.get("/predict")
    assert r.status_code == 200
    data = r.json()
    assert "predicted_regime" in data
    assert data["predicted_regime"] in [0, 1, 2]
    assert "predicted_regime_name" in data
    assert data["predicted_regime_name"] in ["Low Vol", "Mid Vol", "High Vol"]

    total_prob = data["prob_low"] + data["prob_mid"] + data["prob_high"]
    assert abs(total_prob - 1.0) < 0.01, f"Probabilities do not sum to 1: {total_prob}"
    assert data["model_auc"] > 0.65


def test_market_summary_keys():
    """GET /market-summary returns hero section KPIs."""
    r = client.get("/market-summary")
    assert r.status_code == 200
    data = r.json()
    required = [
        "current_regime",
        "garch_vol_current",
        "nifty_close",
        "backtest_sharpe",
        "backtest_max_dd",
        "model_auc",
        "data_end_date",
    ]
    for key in required:
        assert key in data, f"Missing key in market-summary: {key}"

    assert data["current_regime"] in ["Low Vol", "Mid Vol", "High Vol"]
    assert data["nifty_close"] > 0
    assert data["model_auc"] > 0.65


def test_backtest_summary_endpoint():
    """GET /backtest-summary returns strategy performance table."""
    r = client.get("/backtest-summary")
    assert r.status_code == 200
    data = r.json()
    assert "Sharpe Ratio" in data
    assert "Max Drawdown (%)" in data
