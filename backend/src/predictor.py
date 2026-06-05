"""
predictor.py
------------
Loads the trained pipeline (preprocessor + model) and exposes a clean
predict() interface that returns labels with confidence scores.

Returns
-------
{
  "label":       "fake" | "real",
  "confidence":  0.0–1.0,           # probability for the predicted class
  "probabilities": {"fake": float, "real": float},
  "verdict":     "Likely Fake" | "Possibly Fake" | "Uncertain" | ...
}
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# ── confidence thresholds → human-readable verdict ───────────────────────────
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


def _verdict(label: str, confidence: float) -> str:
    thresholds = _FAKE_THRESHOLDS if label == "fake" else _REAL_THRESHOLDS
    for cutoff, text in thresholds:
        if confidence >= cutoff:
            return text
    return _UNCERTAIN_VERDICT


# ── predictor class ───────────────────────────────────────────────────────────

class FakeNewsPredictor:
    """
    Wraps preprocessor + classifier into a single predict() call.

    Usage
    -----
    >>> predictor = FakeNewsPredictor.load()
    >>> result = predictor.predict("Vaccines cause autism, say experts")
    >>> result["label"]       # 'fake'
    >>> result["confidence"]  # 0.94
    >>> result["verdict"]     # 'Almost certainly fake'
    """

    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model        = model
        self.classes_     = preprocessor.classes_   # ['fake', 'real'] or ['real', 'fake']

    # ── load ──────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        preprocessor_path: str | Path | None = None,
        model_path:        str | Path | None = None,
    ) -> "FakeNewsPredictor":
        from src.preprocessor import TextPreprocessor

        pp_path    = Path(preprocessor_path or MODELS_DIR / "preprocessor.joblib")
        model_path = Path(model_path        or MODELS_DIR / "best_model.joblib")

        preprocessor = TextPreprocessor.load(pp_path)
        model        = joblib.load(model_path)
        print(f"[predictor] Loaded preprocessor ← {pp_path}")
        print(f"[predictor] Loaded model        ← {model_path}")
        return cls(preprocessor, model)

    # ── single prediction ─────────────────────────────────────────────────────

    def predict(self, title: str = "", text: str = "") -> dict:
        """
        Predict whether a single article is fake or real.

        Parameters
        ----------
        title : str   Article headline (optional but recommended)
        text  : str   Article body (optional but recommended)

        Returns
        -------
        dict with keys: label, confidence, probabilities, verdict
        """
        df = pd.DataFrame([{"title": title, "text": text}])
        X  = self.preprocessor.transform(df)

        proba  = self.model.predict_proba(X)[0]          # shape (2,)
        label  = self.classes_[int(np.argmax(proba))]
        conf   = float(np.max(proba))

        prob_dict = {cls: round(float(p), 4)
                     for cls, p in zip(self.classes_, proba)}

        return {
            "label":         label,
            "confidence":    round(conf, 4),
            "probabilities": prob_dict,
            "verdict":       _verdict(label, conf),
        }

    # ── batch prediction ──────────────────────────────────────────────────────

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on a DataFrame that has 'title' and/or 'text' columns.
        Adds columns: pred_label, confidence, verdict.
        """
        X     = self.preprocessor.transform(df)
        proba = self.model.predict_proba(X)
        preds = self.classes_[np.argmax(proba, axis=1)]
        confs = np.max(proba, axis=1)

        out = df.copy()
        out["pred_label"] = preds
        out["confidence"] = np.round(confs, 4)
        out["verdict"]    = [_verdict(l, c) for l, c in zip(preds, confs)]
        return out

    # ── explain (top features) ────────────────────────────────────────────────

    def top_features(self, title: str = "", text: str = "", top_n: int = 10) -> list[dict]:
        """
        Return the top TF-IDF features that drove the prediction.
        Works for Logistic Regression and LinearSVC-based models.
        """
        from src.preprocessor import combine_and_clean

        df      = pd.DataFrame([{"title": title, "text": text}])
        cleaned = combine_and_clean(df)
        tfidf   = self.preprocessor.vectorizer
        X       = tfidf.transform(cleaned)

        # Attempt to extract feature importances
        model = self.model
        # Unwrap CalibratedClassifierCV
        if hasattr(model, "calibrated_classifiers_"):
            inner = model.calibrated_classifiers_[0].estimator
        else:
            inner = model

        feature_names = np.array(tfidf.get_feature_names_out())
        x_arr         = X.toarray()[0]
        nonzero_idx   = np.where(x_arr > 0)[0]

        if hasattr(inner, "coef_"):
            # LR / SVC: use coefficient × tfidf weight
            coef = inner.coef_[0][nonzero_idx]
        elif hasattr(inner, "feature_importances_"):
            coef = inner.feature_importances_[nonzero_idx]
        else:
            # Fallback: just return top TF-IDF terms
            coef = x_arr[nonzero_idx]

        scores = coef * x_arr[nonzero_idx]
        order  = np.argsort(np.abs(scores))[::-1][:top_n]

        return [
            {
                "word":   feature_names[nonzero_idx[i]],
                "score":  round(float(scores[i]), 4),
                "tfidf":  round(float(x_arr[nonzero_idx[i]]), 4),
                "direction": "fake" if scores[i] > 0 else "real",
            }
            for i in order
        ]


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        {
            "title": "Scientists confirm mRNA vaccines are safe and effective",
            "text":  "A new meta-analysis covering 40 clinical trials published "
                     "in The Lancet confirms the safety profile of mRNA vaccines.",
        },
        {
            "title": "BOMBSHELL: 5G towers secretly controlling minds — government cover-up EXPOSED",
            "text":  "Whistleblowers reveal shocking truth. Share before deleted!!!",
        },
    ]

    predictor = FakeNewsPredictor.load()
    for s in samples:
        r = predictor.predict(s["title"], s["text"])
        print(f"\nTitle : {s['title'][:70]}")
        print(f"Label : {r['label']}  ({r['confidence']*100:.1f}%)")
        print(f"Verdict: {r['verdict']}")
        print(f"Probs : {r['probabilities']}")
