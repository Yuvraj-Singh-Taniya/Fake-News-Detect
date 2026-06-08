import joblib
import numpy as np
import re
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_FAKE_THRESHOLDS = [
    (0.90, "Almost certainly fake"),
    (0.75, "Very likely fake"),
    (0.60, "Likely fake"),
    (0.50, "Possibly fake"),
]
_REAL_THRESHOLDS = [
    (0.90, "Almost certainly real"),
    (0.75, "Very likely real"),
    (0.60, "Likely real"),
    (0.50, "Possibly real"),
]
_UNCERTAIN_VERDICT = "Uncertain — manual review recommended"


def _verdict(label, confidence):
    if confidence < 0.88:
        return "Uncertain — manual review recommended"
    thresholds = _FAKE_THRESHOLDS if label == "fake" else _REAL_THRESHOLDS
    for cutoff, text in thresholds:
        if confidence >= cutoff:
            return text
    return _UNCERTAIN_VERDICT


def _clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _custom_features(title, text):
    combined = str(title) + " " + str(text)
    return csr_matrix([[
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
    ]], dtype=float)


class FakeNewsPredictor:

    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model      = model
        self.classes_   = model.classes_

    @classmethod
    def load(cls, vectorizer_path=None, model_path=None):
        vp = Path(vectorizer_path or MODELS_DIR / "vectorizer.joblib")
        mp = Path(model_path      or MODELS_DIR / "best_model.joblib")
        vectorizer = joblib.load(vp)
        model      = joblib.load(mp)
        print(f"[predictor] Loaded vectorizer ← {vp}")
        print(f"[predictor] Loaded model      ← {mp}")
        return cls(vectorizer, model)

    def predict(self, title="", text=""):
        content = _clean(title + " " + text)
        tfidf   = self.vectorizer.transform([content])
        custom  = _custom_features(title, text)
        X       = hstack([tfidf, custom])
        proba   = self.model.predict_proba(X)[0]
        label   = self.classes_[int(np.argmax(proba))]
        conf    = float(np.max(proba))
        return {
            "label":         label,
            "confidence":    round(conf, 4),
            "probabilities": {c: round(float(p), 4)
                              for c, p in zip(self.classes_, proba)},
            "verdict":       _verdict(label, conf),
        }

    def predict_batch(self, df):
        import pandas as pd
        titles   = df.get("title", pd.Series([""] * len(df))).fillna("").values
        texts    = df.get("text",  pd.Series([""] * len(df))).fillna("").values
        contents = [_clean(t + " " + b) for t, b in zip(titles, texts)]
        tfidf    = self.vectorizer.transform(contents)
        custom   = csr_matrix(np.array([
            _custom_features(t, b).toarray()[0]
            for t, b in zip(titles, texts)
        ], dtype=float))
        X     = hstack([tfidf, custom])
        proba = self.model.predict_proba(X)
        preds = self.classes_[np.argmax(proba, axis=1)]
        confs = np.max(proba, axis=1)
        out   = df.copy()
        out["pred_label"] = preds
        out["confidence"] = np.round(confs, 4)
        out["verdict"]    = [_verdict(l, c) for l, c in zip(preds, confs)]
        return out

    def top_features(self, title="", text="", top_n=10):
        content       = _clean(title + " " + text)
        tfidf         = self.vectorizer.transform([content])
        custom        = _custom_features(title, text)
        X             = hstack([tfidf, custom])
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        x_arr         = tfidf.toarray()[0]
        nonzero_idx   = np.where(x_arr > 0)[0]

        # get coefficients from inner LR estimator
        inner = self.model.estimators_[0]
        if hasattr(inner, "coef_"):
            coef = inner.coef_[0][nonzero_idx]
        else:
            coef = x_arr[nonzero_idx]

        scores = coef * x_arr[nonzero_idx]
        order  = np.argsort(np.abs(scores))[::-1][:top_n]

        return [
            {
                "word":      feature_names[nonzero_idx[i]],
                "score":     round(float(scores[i]), 4),
                "tfidf":     round(float(x_arr[nonzero_idx[i]]), 4),
                "direction": "fake" if scores[i] > 0 else "real",
            }
            for i in order
        ]