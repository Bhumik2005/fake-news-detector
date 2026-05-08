import re
import logging
import joblib
import feedparser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from sklearn.metrics.pairwise import cosine_similarity

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detector API", version="2.0")

# ─── Load model ─────────────────────────────────────────────────────────────
try:
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    logger.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


# ─── Schema ──────────────────────────────────────────────────────────────────
class NewsInput(BaseModel):
    text: str

    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Input text cannot be empty.")
        return v.strip()


# ─── Preprocessing ───────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Lowercase, remove URLs, strip special chars — must match train.py pipeline."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # remove special chars
    text = re.sub(r"\s+", " ", text).strip()           # collapse whitespace
    return text


# ─── News fetching ───────────────────────────────────────────────────────────
def clean_query(text: str) -> str:
    """Extract a short keyword query from raw input."""
    words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    return " ".join(words[:6])


def fetch_related_news(query: str) -> list:
    try:
        query_encoded = query.replace(" ", "+")
        url = (
            f"https://news.google.com/rss/search"
            f"?q={query_encoded}&hl=en-IN&gl=IN&ceid=IN:en"
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
        logger.info(f"Fetched {len(articles)} articles for query: '{query}'")
        return articles
    except Exception as e:
        logger.warning(f"News fetch failed: {e}")
        return []


# ─── Similarity ──────────────────────────────────────────────────────────────
def compute_best_similarity(user_text: str, articles: list) -> tuple[float, dict | None]:
    """
    Returns (best_score, best_article).
    Hybrid: 60% keyword overlap + 40% TF-IDF cosine similarity.
    """
    if not articles:
        return 0.0, None

    user_clean = preprocess(user_text)
    user_words = set(re.findall(r"\w+", user_clean))
    best_score = 0.0
    best_article = None

    for article in articles:
        article_text = preprocess(
            article["title"] + " " + article["description"]
        )
        article_words = set(re.findall(r"\w+", article_text))

        # Keyword overlap score
        common = user_words.intersection(article_words)
        keyword_score = len(common) / (len(user_words) + 1e-9)

        # TF-IDF cosine similarity
        vecs = vectorizer.transform([user_clean, article_text])
        ml_score = float(cosine_similarity(vecs[0], vecs[1])[0][0])

        score = 0.6 * keyword_score + 0.4 * ml_score

        if score > best_score:
            best_score = score
            best_article = article

    return best_score, best_article


# ─── Fusion logic ────────────────────────────────────────────────────────────
def fuse_verdict(
    ml_label: int,
    ml_fake_prob: float,
    ml_real_prob: float,
    similarity: float,
    has_news: bool,
) -> tuple[str, str, float]:
    """
    Combine ML model prediction + news similarity into one final verdict.

    Weights:
      - ML confidence contributes 50%
      - News similarity contributes 50%

    Returns: (verdict, reason, final_confidence)
    """
    # ML signal: score from 0 (very fake) to 1 (very real)
    ml_signal = ml_real_prob  # already a probability

    # News signal: normalise similarity to [0, 1]
    news_signal = min(similarity, 1.0)

    if not has_news:
        # No news found — rely entirely on ML
        combined = ml_signal
        weight_note = "based on model only (no news found)"
    else:
        # Weighted fusion
        combined = 0.5 * ml_signal + 0.5 * news_signal
        weight_note = "model + news verification"

    # ── Verdict thresholds ────────────────────────────────────────────────────
    if combined >= 0.65:
        verdict = "Real"
        reason = f"Verified by {weight_note}: strong evidence this is real."
    elif combined >= 0.40:
        verdict = "Unverified"
        reason = f"Mixed signals from {weight_note}. Treat with caution."
    else:
        verdict = "Likely Fake"
        reason = f"Low credibility score from {weight_note}."

    # Override: if ML is very confident fake AND similarity is low → Fake
    if ml_fake_prob > 0.80 and similarity < 0.25:
        verdict = "Likely Fake"
        reason = "Model is highly confident this is fake, and no news evidence found."

    # Override: if ML is very confident real AND strong news match → Real
    if ml_real_prob > 0.80 and similarity > 0.55:
        verdict = "Real"
        reason = "Model is highly confident, strongly corroborated by current news."

    return verdict, reason, round(combined * 100, 2)


# ─── Main endpoint ───────────────────────────────────────────────────────────
@app.post("/predict")
def predict(news: NewsInput):
    text = news.text  # already validated + stripped by Pydantic

    # Short input warning
    word_count = len(text.split())
    if word_count < 5:
        return {
            "warning": "Input too short for reliable prediction.",
            "word_count": word_count,
        }

    # ── ML prediction ─────────────────────────────────────────────────────────
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    ml_label = int(model.predict(vec)[0])
    probs = model.predict_proba(vec)[0]
    ml_fake_prob = float(probs[0])
    ml_real_prob = float(probs[1])

    logger.info(
        f"ML → label={ml_label} | fake={ml_fake_prob:.2f} | real={ml_real_prob:.2f}"
    )

    # ── News verification ─────────────────────────────────────────────────────
    query = clean_query(text)
    articles = fetch_related_news(query)
    similarity, best_article = compute_best_similarity(text, articles)

    logger.info(f"News similarity={similarity:.3f} | articles={len(articles)}")

    # ── Fuse both signals ─────────────────────────────────────────────────────
    verdict, reason, confidence = fuse_verdict(
        ml_label=ml_label,
        ml_fake_prob=ml_fake_prob,
        ml_real_prob=ml_real_prob,
        similarity=similarity,
        has_news=len(articles) > 0,
    )

    logger.info(f"Final verdict={verdict} | confidence={confidence}")

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


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}