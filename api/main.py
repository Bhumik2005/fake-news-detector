from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_news
from src.news_fetcher import fetch_news   # ✅ NEW

app = FastAPI()

class NewsRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Fake News Detector API"}

@app.post("/predict")
def predict(request: NewsRequest):
    result = predict_news(request.text)

    # ✅ Fetch live news
    articles = fetch_news(request.text)

    return {
        "prediction": result["prediction"],
        "confidence_fake": result["confidence_fake"],
        "confidence_real": result["confidence_real"],
        "related_articles": articles   # ✅ NEW
    }