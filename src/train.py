import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import SGDClassifier

# Load datasets
fake = pd.read_csv("data/fake.csv", low_memory=False)
true = pd.read_csv("data/true.csv", low_memory=False)

# Assign labels
fake["label"] = 0
true["label"] = 1

# Combine
df = pd.concat([fake, true], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 🔥 Balance dataset
min_count = min(df["label"].value_counts())
df = df.groupby("label").sample(min_count, random_state=42)

print("Class Distribution:\n", df["label"].value_counts())

# Features & target
X = df["text"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 Improved vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 3)   # BIG improvement
)


X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🔥 Improved model
model = SGDClassifier(loss="log_loss")

# Train
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Model and vectorizer saved!")