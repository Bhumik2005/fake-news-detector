"""
explain.py — Explainability layer for the Fake News Detector.

Uses LIME (Local Interpretable Model-agnostic Explanations) to highlight
which words pushed the model toward Fake or Real.

Install: pip install lime
"""

import re
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


def get_lime_explanation(
    text: str,
    pipeline,
    num_features: int = 10,
    num_samples: int = 500,
) -> Optional[dict]:
    """
    Generate a LIME explanation for a single news input.

    Args:
        text:         Raw input text (will be preprocessed internally).
        pipeline:     Fitted sklearn Pipeline (tfidf + clf).
        num_features: How many words to highlight.
        num_samples:  LIME perturbation samples (higher = more accurate, slower).

    Returns:
        {
          "top_fake_words":  [{"word": str, "weight": float}, ...],
          "top_real_words":  [{"word": str, "weight": float}, ...],
          "highlighted_html": str,   # HTML with inline-highlighted spans
        }
        or None on failure.
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        logger.error("LIME not installed. Run: pip install lime")
        return None

    try:
        explainer = LimeTextExplainer(class_names=["Fake", "Real"])

        # LIME needs a predict_proba function that takes a list of strings
        def predict_proba(texts):
            return pipeline.predict_proba(texts)

        explanation = explainer.explain_instance(
            text,
            predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=[0, 1],
        )

        # ── Extract word weights ──────────────────────────────────────────────
        # LIME returns (word, weight_for_class_1) for each feature.
        # Positive weight → pushes toward Real
        # Negative weight → pushes toward Fake

        word_weights = explanation.as_list(label=1)  # weights for "Real" class

        fake_words = [
            {"word": w, "weight": round(abs(wt), 4)}
            for w, wt in word_weights if wt < 0
        ]
        real_words = [
            {"word": w, "weight": round(wt, 4)}
            for w, wt in word_weights if wt > 0
        ]

        # Sort by magnitude
        fake_words.sort(key=lambda x: x["weight"], reverse=True)
        real_words.sort(key=lambda x: x["weight"], reverse=True)

        # ── Build highlighted HTML ────────────────────────────────────────────
        highlighted_html = build_highlighted_html(text, word_weights)

        return {
            "top_fake_words": fake_words[:5],
            "top_real_words": real_words[:5],
            "highlighted_html": highlighted_html,
        }

    except Exception as e:
        logger.warning(f"LIME explanation failed: {e}")
        return None


def build_highlighted_html(text: str, word_weights: list) -> str:
    """
    Wrap words in the original text with colored <span> tags.

    Red   = pushed toward Fake
    Green = pushed toward Real
    Opacity scaled by weight magnitude.
    """
    # Build lookup: word → weight
    weight_map = {w.lower(): wt for w, wt in word_weights}

    tokens = re.split(r"(\s+)", text)  # preserve whitespace
    html_parts = []

    for token in tokens:
        clean = re.sub(r"[^a-z0-9]", "", token.lower())
        if clean in weight_map:
            wt = weight_map[clean]
            magnitude = min(abs(wt) * 8, 1.0)   # scale opacity
            alpha = round(0.2 + 0.6 * magnitude, 2)

            if wt < 0:
                # Fake signal → red
                color = f"rgba(220, 53, 69, {alpha})"
                title = f"Fake signal (−{abs(wt):.3f})"
            else:
                # Real signal → green
                color = f"rgba(25, 135, 84, {alpha})"
                title = f"Real signal (+{wt:.3f})"

            html_parts.append(
                f'<span style="background:{color};border-radius:3px;'
                f'padding:1px 3px;font-weight:500;" title="{title}">'
                f"{token}</span>"
            )
        else:
            html_parts.append(token)

    return "".join(html_parts)


def get_tfidf_top_features(
    text: str,
    pipeline,
    top_n: int = 10,
) -> dict:
    """
    Lightweight fallback when LIME is unavailable or too slow.
    Uses raw TF-IDF feature scores to find the most important words.

    Returns:
        {
          "top_words": [{"word": str, "score": float}, ...]
        }
    """
    try:
        vectorizer = pipeline.named_steps["tfidf"]
        vec = vectorizer.transform([text])

        feature_names = vectorizer.get_feature_names_out()
        scores = vec.toarray()[0]

        # Get indices of top scoring features
        top_indices = np.argsort(scores)[::-1][:top_n]

        top_words = [
            {"word": feature_names[i], "score": round(float(scores[i]), 4)}
            for i in top_indices
            if scores[i] > 0
        ]

        return {"top_words": top_words}

    except Exception as e:
        logger.warning(f"TF-IDF feature extraction failed: {e}")
        return {"top_words": []}