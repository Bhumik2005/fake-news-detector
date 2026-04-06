🚀 Fake News Detection System

An end-to-end Machine Learning project that detects whether a news article is Fake or Real using NLP techniques, deployed with a full backend and frontend, and enhanced with real-time news verification.

📌 Features
🧠 Machine Learning model for fake news detection
🔤 NLP using TF-IDF vectorization (n-grams)
⚡ Backend API built with FastAPI
🎨 Interactive UI using Streamlit
🌐 Real-time news verification via NewsAPI
📊 Confidence scores for predictions
⚠️ Warning for short/low-information inputs

🏗️ Project Structure

```
fake-news-detector/
│
├── app/
│   └── streamlit_app.py      # Frontend UI
│
├── api/
│   └── main.py               # FastAPI backend
│
├── src/
│   ├── train.py              # Model training
│   ├── predict.py            # Prediction logic
│   └── news_fetcher.py       # Live news API integration
│
├── data/
│   ├── fake.csv
│   └── true.csv
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── requirements.txt
└── README.md
```

⚙️ How It Works
1. User inputs a news article
2. Text is transformed using TF-IDF vectorization
3. ML model predicts Fake or Real
4. Confidence scores are generated
5. Related articles are fetched using News API
6. Results are displayed in the UI

🧪 Model Details
1. Algorithm: Logistic Regression / SGD Classifier
2. Feature Extraction: TF-IDF (with n-grams)
3. Dataset: Fake + Real news CSV files
4. Balanced dataset for improved performance

🚀 How to Run Locally

1️⃣ Clone the repository
```
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

2️⃣ Create virtual environment
```
python -m venv ml_env
ml_env\Scripts\activate   # Windows
```
3️⃣ Install dependencies
```
pip install -r requirements.txt
```

4️⃣ Train the model
```
python src/train.py
```
5️⃣ Run backend
```
uvicorn api.main:app --reload
```
6️⃣ Run frontend
```
streamlit run app/streamlit_app.py
```

🔑 API Key Setup

To enable live news fetching:

1. Get API key from NewsAPI
2. Add it in:
   ```
   API_KEY = "your_api_key_here"
   ```

📊 Example Output
1. Prediction: Fake / Real
2. Confidence Scores
3. Related News Articles with sources

🌐 Future Improvements
1. 🔍 Explainable AI (highlight important words)
2. 🧠 Upgrade to BERT-based models
3. ⭐ Source credibility scoring
4. ☁️ Cloud deployment (AWS / Render)
5. 🗄️ Store predictions in database

💡 Learnings
1. End-to-end ML pipeline development
2. NLP preprocessing and feature engineering
3. API development and integration
4. Frontend + backend integration
5. Real-world problem solving


👨‍💻 Author

Bhumik Kumta
