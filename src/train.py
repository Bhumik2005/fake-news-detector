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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

logger.info("Loading datasets...")
fake = pd.read_csv("data/fake.csv", low_memory=False)
true = pd.read_csv("data/true.csv", low_memory=False)

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

min_count = df["label"].value_counts().min()
df = df.groupby("label").sample(min_count, random_state=42).reset_index(drop=True)
logger.info(f"Balanced class distribution:\n{df['label'].value_counts().to_string()}")

df["content"] = (
    df.get("title", pd.Series([""] * len(df))).fillna("")
    + " "
    + df["text"].fillna("")
)

logger.info("Preprocessing text...")
df["content"] = df["content"].apply(preprocess)

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=3,
    )),
    ("clf", SGDClassifier(
        loss="log_loss",
        max_iter=100,
        random_state=42,
        n_jobs=-1,
    )),
])

logger.info("Running 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
logger.info(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

logger.info("Training final model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

logger.info(f"Test Accuracy: {test_accuracy:.4f}")
logger.info(f"ROC-AUC: {roc_auc:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Fake News Detector", fontsize=13)
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Fake News Detector", fontsize=13)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("reports/roc_curve.png", dpi=150)
plt.close()

metrics = {
    "test_accuracy": round(test_accuracy, 4),
    "roc_auc": round(roc_auc, 4),
    "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
    "cv_std_accuracy": round(float(cv_scores.std()), 4),
    "cv_fold_scores": [round(s, 4) for s in cv_scores.tolist()],
    "train_size": len(X_train),
    "test_size": len(X_test),
}
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(pipeline, "models/pipeline.pkl")
joblib.dump(pipeline.named_steps["clf"], "models/model.pkl")
joblib.dump(pipeline.named_steps["tfidf"], "models/vectorizer.pkl")

print("\n✅ Training complete.")
print(f"   Accuracy : {test_accuracy:.4f}")
print(f"   ROC-AUC  : {roc_auc:.4f}")
print(f"   CV Score : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("\nArtifacts saved in reports/")