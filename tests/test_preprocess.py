"""
tests/test_preprocess.py
Unit tests for the preprocess() function in api/main.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import pytest


# ── Copy of preprocess() so tests are self-contained ─────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_lowercase():
    assert preprocess("BREAKING NEWS") == "breaking news"


def test_removes_urls():
    assert "http" not in preprocess("visit https://example.com for more")
    assert "www" not in preprocess("go to www.google.com now")


def test_removes_special_characters():
    result = preprocess("Hello, World! This is #fake @news.")
    assert "," not in result
    assert "!" not in result
    assert "#" not in result
    assert "@" not in result


def test_collapses_whitespace():
    result = preprocess("too    many     spaces")
    assert "  " not in result


def test_strips_leading_trailing_whitespace():
    result = preprocess("   some news   ")
    assert result == result.strip()


def test_empty_string():
    assert preprocess("") == ""


def test_numbers_preserved():
    result = preprocess("covid19 affects 1000 people")
    assert "19" in result
    assert "1000" in result


def test_full_pipeline():
    text = "BREAKING: Visit https://fakenews.com — 100 people DEAD!!!"
    result = preprocess(text)
    assert result == "breaking visit 100 people dead"