from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import requests
import os
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load model
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# 🔐 Secure API key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


class NewsInput(BaseModel):
    text: str


# 🔧 Clean query for better search
def clean_query(text):
    words = text.split()
    return " ".join(words[:6])  # take first 5–6 words


# 🔎 Fetch live news (IMPROVED)
def fetch_related_news(query):
    try:
        url = "https://newsapi.org/v2/everything"

        params = {
            "q": clean_query(query),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "apiKey": NEWS_API_KEY
        }

        res = requests.get(url, params=params)
        data = res.json()

        print("NEWS API RESPONSE:", data)  # 🔥 DEBUG

        articles = []

        if "articles" in data:
            for a in data["articles"]:
                articles.append({
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "url": a.get("url", ""),
                    "source": a["source"].get("name", "Unknown"),
                    "publishedAt": a.get("publishedAt", "")
                })

        return articles[:5]

    except Exception as e:
        print("ERROR FETCHING NEWS:", str(e))
        return []


# 🧠 Similarity check
def compute_similarity(user_text, articles):
    if not articles:
        return 0.0

    texts = [user_text] + [
        a["title"] + " " + a["description"]
        for a in articles
    ]

    vectors = vectorizer.transform(texts)

    sims = cosine_similarity(vectors[0], vectors[1:])
    return float(max(sims[0]))


# 🚀 MAIN API
@app.post("/predict")
def predict(news: NewsInput):
    text = news.text.strip()

    if not text:
        return {"error": "Empty input"}

    # ML prediction
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]

    # Fetch news
    articles = fetch_related_news(text)

    # Similarity
    similarity = compute_similarity(text, articles)

    # 🔥 FINAL DECISION (IMPROVED)
    if similarity > 0.6:
        final = "Real"
        reason = "Strong match with latest news"

    elif similarity > 0.3:
        final = "Unverified"
        reason = "Partial match with current news"

    else:
        if articles:
            final = "Likely Fake"
            reason = "No strong match in news sources"
        else:
            final = "Unverified"
            reason = "No recent news found"

    return {
        "prediction": final,
        "match_percentage": round(similarity * 100, 2),
        "reason": reason,
        "related_articles": articles
    }