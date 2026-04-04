import requests

API_KEY = "e43e423d76214d629ce59e7441755647"

def fetch_news(query):
    url = "https://newsapi.org/v2/everything"
    
    params = {
        "q": query,
        "apiKey": API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return []

    data = response.json()
    articles = data.get("articles", [])

    results = []

    for article in articles:
        results.append({
            "title": article["title"],
            "source": article["source"]["name"],
            "url": article["url"]
        })

    return results