from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# 🔑 PUT YOUR API KEY HERE
NEWS_API_KEY = "e43e423d76214d629ce59e7441755647"

class NewsInput(BaseModel):
    text: str


# 🔎 Fetch live news
def fetch_related_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    res = requests.get(url).json()

    articles = []
    if "articles" in res:
        for a in res["articles"][:5]:
            articles.append({
                "title": a["title"],
                "description": a["description"] or "",
                "url": a["url"],
                "source": a["source"]["name"]
            })
    return articles


# 🧠 Similarity check
def compute_similarity(user_text, articles):
    if not articles:
        return 0.0

    texts = [user_text] + [
        (a["title"] or "") + " " + (a["description"] or "")
        for a in articles
    ]

    vectors = vectorizer.transform(texts)

    user_vec = vectors[0]
    news_vecs = vectors[1:]

    sims = cosine_similarity(user_vec, news_vecs)
    return float(max(sims[0]))


# 🚀 MAIN PREDICTION API
@app.post("/predict")
def predict(news: NewsInput):
    text = news.text.strip()

    # Handle empty input
    if not text:
        return {"error": "Empty input"}

    # STEP 1: Detect question
    is_question = (
        "?" in text or
        text.lower().startswith(("is", "are", "was", "were", "did", "does"))
    )

    # STEP 2: ML prediction
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]  # 0 = Fake, 1 = Real
    probs = model.predict_proba(vec)[0]

    # STEP 3: Fetch live news
    articles = fetch_related_news(text)

    # STEP 4: Similarity score
    similarity = compute_similarity(text, articles)

    # STEP 5: FINAL DECISION LOGIC 🔥
    if similarity > 0.5:
        final = "Real"
        reason = "Matched with real news articles"
    elif similarity < 0.2 and pred == 0:
        final = "Fake"
        reason = "No matching news + model prediction"
    else:
        final = "Uncertain"
        reason = "Not enough reliable information"

    return {
        "prediction": final,
        "confidence_real": float(probs[1]),
        "confidence_fake": float(probs[0]),
        "similarity_score": similarity,
        "reason": reason,
        "related_articles": articles
    }