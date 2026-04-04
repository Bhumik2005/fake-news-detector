import joblib

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_news(text):
    text_vec = vectorizer.transform([text])
    
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]

    # ✅ DEBUG PRINT (correct place)
    print("Prediction probabilities:", probability)

    return {
        "prediction": int(prediction),
        "confidence_fake": float(probability[0]),
        "confidence_real": float(probability[1])
    }