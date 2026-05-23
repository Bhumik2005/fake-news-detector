"""
tests/test_predict.py
Unit tests for model prediction output.
Loads the real pipeline and checks output structure and types.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import pytest
import joblib


# ── Load pipeline once for all tests ─────────────────────────────────────────
@pytest.fixture(scope="module")
def pipeline():
    path = os.path.join(os.path.dirname(__file__), "..", "models", "pipeline.pkl")
    return joblib.load(path)


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_prediction_is_binary(pipeline):
    """Model should predict 0 (Fake) or 1 (Real) only."""
    result = pipeline.predict([preprocess("The president signed a new bill today")])
    assert result[0] in [0, 1]


def test_probabilities_sum_to_one(pipeline):
    """Fake + Real probabilities must sum to 1.0."""
    probs = pipeline.predict_proba([preprocess("Scientists discover water on Mars")])[0]
    assert abs(sum(probs) - 1.0) < 1e-6


def test_probabilities_between_zero_and_one(pipeline):
    """Each probability must be in [0, 1]."""
    probs = pipeline.predict_proba([preprocess("Aliens have landed in New York")])[0]
    for p in probs:
        assert 0.0 <= p <= 1.0


def test_obvious_real_news(pipeline):
    """Clearly factual news should be predicted as Real (label=1)."""
    text = preprocess(
        "The Federal Reserve raised interest rates by 25 basis points "
        "at its meeting on Wednesday, citing continued inflation concerns."
    )
    result = pipeline.predict([text])[0]
    assert result == 1


def test_obvious_fake_news(pipeline):
    """Clearly fake/conspiracy news should be predicted as Fake (label=0)."""
    text = preprocess(
        "SHOCKING: Government admits chemtrails are mind control. "
        "Illuminati confirms global microchip implant plan exposed."
    )
    result = pipeline.predict([text])[0]
    assert result == 0


def test_predict_returns_two_probabilities(pipeline):
    """predict_proba should return exactly 2 values: [fake_prob, real_prob]."""
    probs = pipeline.predict_proba([preprocess("Some news text here")])[0]
    assert len(probs) == 2


def test_pipeline_handles_short_input(pipeline):
    """Pipeline should not crash on very short input."""
    result = pipeline.predict([preprocess("Trump")])
    assert result[0] in [0, 1]


def test_pipeline_handles_empty_string(pipeline):
    """Pipeline should not crash on empty string."""
    result = pipeline.predict([""])
    assert result[0] in [0, 1]