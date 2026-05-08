import os
import re
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detector API", version="3.0")

# ─── Load pipeline (vectorizer + model in one object) ────────────────────────
try:
    pipeline = joblib.load("models/pipeline.pkl")
    logger.info("Pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load pipeline: {e}")
    raise


# ─── Schemas ─────────────────────────────────────────────────────────────────
class NewsInput(BaseModel):
    text: str

    @validator("text")
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Input text cannot be empty.")
        return v.strip()


class ExplainInput(BaseModel):
    text: str
    use_lime: bool = True      # set False to use fast TF-IDF fallback

    @validator("text")
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Input text cannot be empty.")
        return v.strip()


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Must stay in sync with src/train.py preprocess()."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── News helpers ─────────────────────────────────────────────────────────────
def clean_query(text: str) -> str:
    words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    return " ".join(words[:6])


def fetch_related_news(query: str) -> list:
    try:
        url = (
            f"https://news.google.com/rss/search"
            f"?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
        )
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:5]:
            articles.append({
                "title": entry.title,
                "description": entry.get("summary", ""),
                "url": entry.link,
                "source": entry.get("source", {}).get("title", "Google News"),
                "publishedAt": entry.get("published", ""),
            })
        logger.info(f"Fetched {len(articles)} articles for: '{query}'")
        return articles
    except Exception as e:
        logger.warning(f"News fetch failed: {e}")
        return []


def compute_best_similarity(user_text: str, articles: list) -> tuple:
    if not articles:
        return 0.0, None

    user_clean = preprocess(user_text)
    user_words = set(re.findall(r"\w+", user_clean))
    best_score, best_article = 0.0, None

    for article in articles:
        article_text = preprocess(article["title"] + " " + article["description"])
        article_words = set(re.findall(r"\w+", article_text))

        keyword_score = len(user_words & article_words) / (len(user_words) + 1e-9)

        vecs = pipeline.named_steps["tfidf"].transform([user_clean, article_text])
        ml_score = float(cosine_similarity(vecs[0], vecs[1])[0][0])

        score = 0.6 * keyword_score + 0.4 * ml_score
        if score > best_score:
            best_score, best_article = score, article

    return best_score, best_article


def fuse_verdict(ml_label, ml_fake_prob, ml_real_prob, similarity, has_news):
    ml_signal = ml_real_prob
    news_signal = min(similarity, 1.0)

    combined = ml_signal if not has_news else 0.5 * ml_signal + 0.5 * news_signal
    weight_note = "model only (no news)" if not has_news else "model + news"

    if combined >= 0.65:
        verdict, reason = "Real", f"Strong credibility from {weight_note}."
    elif combined >= 0.40:
        verdict, reason = "Unverified", f"Mixed signals from {weight_note}."
    else:
        verdict, reason = "Likely Fake", f"Low credibility from {weight_note}."

    if ml_fake_prob > 0.80 and similarity < 0.25:
        verdict = "Likely Fake"
        reason = "Model highly confident fake; no corroborating news found."
    if ml_real_prob > 0.80 and similarity > 0.55:
        verdict = "Real"
        reason = "Model highly confident real; strongly corroborated by news."

    return verdict, reason, round(combined * 100, 2)


# ─── /predict ────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(news: NewsInput):
    text = news.text

    if len(text.split()) < 5:
        return {"warning": "Input too short for reliable prediction."}

    clean = preprocess(text)
    ml_label = int(pipeline.predict([clean])[0])
    probs = pipeline.predict_proba([clean])[0]
    ml_fake_prob, ml_real_prob = float(probs[0]), float(probs[1])

    logger.info(f"ML → label={ml_label} fake={ml_fake_prob:.2f} real={ml_real_prob:.2f}")

    articles = fetch_related_news(clean_query(text))
    similarity, best_article = compute_best_similarity(text, articles)

    logger.info(f"Similarity={similarity:.3f} articles={len(articles)}")

    verdict, reason, confidence = fuse_verdict(
        ml_label, ml_fake_prob, ml_real_prob, similarity, len(articles) > 0
    )

    logger.info(f"Verdict={verdict} confidence={confidence}")

    return {
        "prediction": verdict,
        "confidence_score": confidence,
        "reason": reason,
        "ml_details": {
            "label": "Real" if ml_label == 1 else "Fake",
            "fake_probability": round(ml_fake_prob * 100, 2),
            "real_probability": round(ml_real_prob * 100, 2),
        },
        "news_details": {
            "similarity_score": round(similarity * 100, 2),
            "articles_found": len(articles),
        },
        "best_match": best_article,
        "related_articles": articles,
    }


# ─── /explain ────────────────────────────────────────────────────────────────
@app.post("/explain")
def explain(req: ExplainInput):
    """
    Returns word-level explanation for why the model made its prediction.

    use_lime=True  → full LIME explanation with highlighted HTML (slower ~2s)
    use_lime=False → fast TF-IDF top features (instant)
    """
    text = req.text

    if len(text.split()) < 5:
        return {"warning": "Input too short to explain."}

    clean = preprocess(text)
    ml_label = int(pipeline.predict([clean])[0])
    probs = pipeline.predict_proba([clean])[0]

    if req.use_lime:
        logger.info("Generating LIME explanation...")
        result = get_lime_explanation(text, pipeline)
        method = "lime"
    else:
        logger.info("Generating TF-IDF explanation...")
        result = get_tfidf_top_features(clean, pipeline)
        method = "tfidf"

    if result is None:
        result = get_tfidf_top_features(clean, pipeline)
        method = "tfidf_fallback"

    return {
        "prediction": "Real" if ml_label == 1 else "Fake",
        "fake_probability": round(float(probs[0]) * 100, 2),
        "real_probability": round(float(probs[1]) * 100, 2),
        "explanation_method": method,
        **result,
    }


# ─── /health ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0"}