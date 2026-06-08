import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.sparse import hstack, csr_matrix
import joblib

# ── Text cleaner ───────────────────────────────────────────────────────────
def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Custom features ────────────────────────────────────────────────────────
def extract_features(titles, texts):
    rows = []
    for title, text in zip(titles, texts):
        combined = str(title) + " " + str(text)
        rows.append([
            combined.count("!"),
            combined.count("?"),
            len(re.findall(r'\b[A-Z]{3,}\b', combined)),
            sum(1 for c in combined if c.isupper()) / max(len(combined), 1),
            1 if any(w in combined.lower() for w in
                     ["reuters", "according to", "told reporters",
                      "said in a statement", "confirmed", "published"]) else 0,
            1 if '"' in combined or "said" in combined else 0,
            len(str(title).split()),
            len(str(text).split()),
            1 if re.search(r'\d+', combined) else 0,
        ])
    return csr_matrix(np.array(rows, dtype=float))

# ── Load ISOT ──────────────────────────────────────────────────────────────
def load_isot():
    print("Loading ISOT...")
    fake = pd.read_csv("data/Fake.csv")[["title", "text"]]
    real = pd.read_csv("data/True.csv")[["title", "text"]]
    fake["label"] = "fake"
    real["label"] = "real"
    df = pd.concat([fake, real], ignore_index=True)
    print(f"  ISOT: {len(df)} rows")
    return df

# ── Load FakeNewsNet ───────────────────────────────────────────────────────
def load_fakenewsnet():
    dfs = []
    files = {
        "data/gossipcop_fake.csv": "fake",
        "data/gossipcop_real.csv": "real",
        "data/politifact_fake.csv": "fake",
        "data/politifact_real.csv": "real",
    }
    for path, label in files.items():
        if Path(path).exists():
            try:
                df = pd.read_csv(path)
                # FakeNewsNet uses different column names
                if "title" not in df.columns:
                    df["title"] = df.get("news_title", df.get("headline", ""))
                if "text" not in df.columns:
                    df["text"] = df.get("news_text", df.get("content", ""))
                df = df[["title", "text"]].copy()
                df["label"] = label
                dfs.append(df)
                print(f"  {path}: {len(df)} rows")
            except Exception as e:
                print(f"  Skipping {path}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["title", "text", "label"])

# ── Load LIAR ──────────────────────────────────────────────────────────────
def load_liar():
    # LIAR labels: pants-fire, false, barely-true → fake
    #              half-true, mostly-true, true     → real
    fake_labels = {"pants-fire", "false", "barely-true"}
    real_labels = {"half-true", "mostly-true", "true"}
    dfs = []
    for fname in ["data/liar_train.tsv", "data/liar_test.tsv", "data/liar_valid.tsv"]:
        if Path(fname).exists():
            try:
                df = pd.read_csv(fname, sep="\t", header=None,
                                 usecols=[1, 2],
                                 names=["verdict", "text"])
                df["title"] = df["text"]
                df = df[df["verdict"].isin(fake_labels | real_labels)].copy()
                df["label"] = df["verdict"].apply(
                    lambda x: "fake" if x in fake_labels else "real"
                )
                df = df[["title", "text", "label"]]
                dfs.append(df)
                print(f"  {fname}: {len(df)} rows")
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["title", "text", "label"])

# ── Combine all datasets ───────────────────────────────────────────────────
print("=" * 50)
print("Loading datasets...")
print("=" * 50)

frames = []
frames.append(load_isot())

fnn = load_fakenewsnet()
if len(fnn) > 0:
    frames.append(fnn)

liar = load_liar()
if len(liar) > 0:
    frames.append(liar)

df = pd.concat(frames, ignore_index=True)
df = df.dropna(subset=["label"])
df["title"] = df["title"].fillna("")
df["text"]  = df["text"].fillna("")

print(f"\nTotal before cleaning : {len(df)}")
print(f"Fake: {(df.label=='fake').sum()}  Real: {(df.label=='real').sum()}")

# ── Clean ──────────────────────────────────────────────────────────────────
print("\nCleaning text...")
df["title_clean"]   = df["title"].apply(clean)
df["text_clean"]    = df["text"].apply(clean)
df["content"]       = df["title_clean"] + " " + df["text_clean"]
df = df[df["content"].str.len() > 20].reset_index(drop=True)

print(f"Total after cleaning  : {len(df)}")

# ── Split ──────────────────────────────────────────────────────────────────
X_title = df["title"].values
X_text  = df["text"].values
X_content = df["content"].values
y = df["label"].values

(Xc_train, Xc_test,
 Xt_train, Xt_test,
 Xb_train, Xb_test,
 y_train,  y_test) = train_test_split(
    X_content, X_title, X_text, y,
    test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(Xc_train)}  Test: {len(Xc_test)}")

# ── TF-IDF ─────────────────────────────────────────────────────────────────
print("\nFitting TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=60000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
Xc_train_vec = vectorizer.fit_transform(Xc_train)
Xc_test_vec  = vectorizer.transform(Xc_test)

# ── Custom features ────────────────────────────────────────────────────────
print("Extracting custom features...")
Xf_train = extract_features(Xt_train, Xb_train)
Xf_test  = extract_features(Xt_test,  Xb_test)

X_train_full = hstack([Xc_train_vec, Xf_train])
X_test_full  = hstack([Xc_test_vec,  Xf_test])

# ── Train ensemble ─────────────────────────────────────────────────────────
print("\nTraining ensemble model...")
lr  = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1)
svc = CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=2000))

ensemble = VotingClassifier(
    estimators=[("lr", lr), ("svc", svc)],
    voting="soft",
    n_jobs=-1
)
ensemble.fit(X_train_full, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────
print("\nEvaluating...")
y_pred = ensemble.predict(X_test_full)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, pos_label="fake")

print(f"\nAccuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

# ── Save ───────────────────────────────────────────────────────────────────
Path("models").mkdir(exist_ok=True)
joblib.dump(vectorizer, "models/vectorizer.joblib")
joblib.dump(ensemble,   "models/best_model.joblib")

metrics = {
    "accuracy": round(acc, 4),
    "f1":       round(f1, 4),
    "train_size": len(Xc_train),
    "test_size":  len(Xc_test),
    "datasets":   ["ISOT", "FakeNewsNet", "LIAR"],
}
with open("models/metrics_new.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print("  models/vectorizer.joblib")
print("  models/best_model.joblib")
print("  models/metrics_new.json")