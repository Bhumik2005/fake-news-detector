"""
tests/test_api.py
Integration tests for all FastAPI endpoints.
Uses TestClient — no running server or real database needed.
"""
import sys
import os

# ── Set fake DB URL BEFORE any app imports ────────────────────────────────────
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base, get_db
from api.main import app

# ── Create a real in-memory SQLite DB for tests ───────────────────────────────
TEST_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Override the DB dependency with SQLite
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────
def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_returns_version(client):
    response = client.get("/health")
    assert "version" in response.json()


# ── /predict ──────────────────────────────────────────────────────────────────
def test_predict_returns_200(client):
    response = client.post("/predict", json={"text": "The president signed a new trade deal today"})
    assert response.status_code == 200


def test_predict_response_has_required_fields(client):
    response = client.post("/predict", json={"text": "Scientists confirm vaccine effectiveness in new study"})
    data = response.json()
    assert "prediction" in data
    assert "confidence_score" in data
    assert "reason" in data
    assert "ml_details" in data
    assert "news_details" in data


def test_predict_verdict_is_valid(client):
    response = client.post("/predict", json={"text": "NASA launches new mission to explore Jupiter moons"})
    verdict = response.json()["prediction"]
    assert verdict in ["Real", "Likely Fake", "Unverified"]


def test_predict_confidence_is_percentage(client):
    response = client.post("/predict", json={"text": "Stock markets hit record high amid economic recovery"})
    confidence = response.json()["confidence_score"]
    assert 0 <= confidence <= 100


def test_predict_ml_details_structure(client):
    response = client.post("/predict", json={"text": "Government announces new infrastructure plan worth billions"})
    ml = response.json()["ml_details"]
    assert "label" in ml
    assert "fake_probability" in ml
    assert "real_probability" in ml
    assert ml["label"] in ["Real", "Fake"]


def test_predict_empty_input_returns_error(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_missing_text_field_returns_error(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


# ── /explain ──────────────────────────────────────────────────────────────────
def test_explain_returns_200(client):
    response = client.post("/explain", json={
        "text": "The central bank raised interest rates to combat inflation",
        "use_lime": False
    })
    assert response.status_code == 200


def test_explain_response_has_prediction(client):
    response = client.post("/explain", json={
        "text": "Scientists discover new treatment for cancer",
        "use_lime": False
    })
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["Real", "Fake"]


def test_explain_has_probabilities(client):
    response = client.post("/explain", json={
        "text": "World leaders meet to discuss climate change agreement",
        "use_lime": False
    })
    data = response.json()
    assert "fake_probability" in data
    assert "real_probability" in data


# ── /history ──────────────────────────────────────────────────────────────────
def test_history_returns_200(client):
    response = client.get("/history")
    assert response.status_code == 200


def test_history_returns_list(client):
    response = client.get("/history")
    assert isinstance(response.json(), list)


# ── /stats ────────────────────────────────────────────────────────────────────
def test_stats_returns_200(client):
    response = client.get("/stats")
    assert response.status_code == 200


def test_stats_has_required_fields(client):
    response = client.get("/stats")
    data = response.json()
    assert "total_predictions" in data
    assert "verdict_breakdown" in data
    assert "average_confidence" in data