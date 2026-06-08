import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import re
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake["label"] = "fake"
real["label"] = "real"

df = pd.concat([fake, real], ignore_index=True)
print(f"Total samples: {len(df)} ({len(fake)} fake, {len(real)} real)")

# ── Clean text ─────────────────────────────────────────────────────────────
def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)     # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning text...")
df["title"]   = df["title"].apply(clean)
df["text"]    = df["text"].apply(clean)
df["content"] = df["title"] + " " + df["text"]
df = df[df["content"].str.len() > 20].reset_index(drop=True)

# ── Split ──────────────────────────────────────────────────────────────────
X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ── Vectorize ──────────────────────────────────────────────────────────────
print("Fitting TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── Train ──────────────────────────────────────────────────────────────────
print("Training Logistic Regression...")
model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1)
model.fit(X_train_vec, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, pos_label="fake")

print(f"\nAccuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

# ── Save ───────────────────────────────────────────────────────────────────
Path("models").mkdir(exist_ok=True)
joblib.dump(vectorizer, "models/vectorizer.joblib")
joblib.dump(model,      "models/best_model.joblib")
print("\nSaved models/vectorizer.joblib and models/best_model.joblib")