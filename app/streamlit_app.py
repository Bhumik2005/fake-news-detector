import joblib
import feedparser
import re
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load model
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


class NewsInput(BaseModel):
    text: str


# 🔧 Clean query
def clean_query(text):
    words = text.lower().replace("?", "").split()
    return " ".join(words[:6])


# 🔎 Fetch Google News
def fetch_related_news(query):
    try:
        query = query.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

        feed = feedparser.parse(url)

        articles = []

        for entry in feed.entries[:5]:
            articles.append({
                "title": entry.title,
                "description": entry.get("summary", ""),
                "url": entry.link,
                "source": entry.get("source", {}).get("title", "Google News"),
                "publishedAt": entry.get("published", "")
            })

        return articles

    except Exception as e:
        print("GOOGLE NEWS ERROR:", str(e))
        return []


# 🧠 Hybrid similarity
def compute_similarity(user_text, articles):
    if not articles:
        return 0.0

    user_words = set(re.findall(r"\w+", user_text.lower()))
    scores = []

    for article in articles:
        article_text = (article["title"] + " " + article["description"]).lower()
        article_words = set(re.findall(r"\w+", article_text))

        # Keyword overlap
        common_words = user_words.intersection(article_words)
        keyword_score = len(common_words) / (len(user_words) + 1)

        # ML similarity
        vecs = vectorizer.transform([user_text, article_text])
        ml_score = cosine_similarity(vecs[0], vecs[1])[0][0]

        # Final hybrid score
        final_score = (0.6 * keyword_score) + (0.4 * ml_score)
        scores.append(final_score)

    return max(scores)


# 🔎 Get best matching article
def get_best_article(user_text, articles):
    best_score = 0
    best_article = None

    for article in articles:
        article_text = (article["title"] + " " + article["description"]).lower()
        vecs = vectorizer.transform([user_text, article_text])
        score = cosine_similarity(vecs[0], vecs[1])[0][0]

        if score > best_score:
            best_score = score
            best_article = article

    return best_article


# 🔥 Boost for questions
def boost_for_questions(text, similarity):
    if "?" in text.lower():
        return min(similarity + 0.1, 1.0)
    return similarity


# 🚀 MAIN API
@app.post("/predict")
def predict(news: NewsInput):
    text = news.text.strip()

    if not text:
        return {"error": "Empty input"}

    # ML prediction (kept but not primary)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]

    # Fetch news
    articles = fetch_related_news(clean_query(text))

    # Similarity
    similarity = compute_similarity(text, articles)
    similarity = boost_for_questions(text, similarity)

    # Best article
    best_article = get_best_article(text, articles)

    # Decision logic
    if similarity > 0.6:
        final = "Real"
        reason = "This matches multiple recent news reports"

    elif similarity > 0.3:
        final = "Unverified"
        reason = "Some news mentions found, but not strongly confirmed"

    else:
        if articles:
            final = "Likely Fake"
            reason = "No strong evidence in current news sources"
        else:
            final = "Unverified"
            reason = "No recent news available on this topic"

    return {
        "prediction": final,
        "match_percentage": round(similarity * 100, 2),
        "reason": reason,
        "best_match": best_article,
        "related_articles": articles
    }