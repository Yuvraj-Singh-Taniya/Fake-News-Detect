"""
trainer.py
----------
Trains multiple Scikit-learn classifiers on the pre-vectorised data,
evaluates each one, and persists the best pipeline to disk.

Models trained
--------------
  1. Logistic Regression    (fast, strong baseline)
  2. LinearSVC              (excellent for high-dim text)
  3. Multinomial Naive Bayes (classic text baseline)
  4. Random Forest          (ensemble, interpretable)
  5. Gradient Boosting      (high accuracy, slower)

All classifiers are wrapped in CalibratedClassifierCV so that
predict_proba() returns well-calibrated confidence scores.
"""

import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import Any

from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import LinearSVC
from sklearn.naive_bayes   import MultinomialNB
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration   import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── classifier catalogue ──────────────────────────────────────────────────────

def _build_classifiers() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42,
        ),
        "linear_svc": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42),
            cv=3, method="isotonic",
        ),
        "naive_bayes": MultinomialNB(alpha=0.1),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1,
            max_depth=5, random_state=42,
        ),
    }


# ── evaluation helpers ────────────────────────────────────────────────────────

def evaluate(
    model,
    X: sp.spmatrix,
    y: np.ndarray,
    split_name: str = "val",
) -> dict:
    """Return a dict of evaluation metrics for one model / split."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return {
        "split":     split_name,
        "accuracy":  round(accuracy_score(y, y_pred),  4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred,        zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y, y_prob),  4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


# ── main trainer ──────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Train, compare, and persist fake-news classifiers.

    Usage
    -----
    >>> trainer = ModelTrainer()
    >>> results = trainer.train_all(X_train, y_train, X_val, y_val)
    >>> trainer.evaluate_best(X_test, y_test)
    >>> trainer.save_best()
    """

    def __init__(self):
        self.classifiers  = _build_classifiers()
        self.trained_     = {}       # name → fitted model
        self.metrics_     = {}       # name → metrics dict
        self.best_name_   = None
        self.best_model_  = None

    # ── training ──────────────────────────────────────────────────────────────

    def train_all(
        self,
        X_train: sp.spmatrix,
        y_train: np.ndarray,
        X_val:   sp.spmatrix,
        y_val:   np.ndarray,
        cv_folds: int = 5,
    ) -> dict:
        """
        Fit every classifier, collect val metrics, pick best by F1.
        Returns dict of all metrics.
        """
        print(f"\n{'='*55}")
        print(f"  Training {len(self.classifiers)} classifiers …")
        print(f"{'='*55}")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for name, clf in self.classifiers.items():
            print(f"\n[trainer] {name} …", flush=True)

            # Handle models that don't natively support predict_proba
            if not hasattr(clf, "predict_proba"):
                clf = CalibratedClassifierCV(clf, cv=3)

            clf.fit(X_train, y_train)
            self.trained_[name] = clf

            val_metrics = evaluate(clf, X_val, y_val, split_name="val")
            cv_scores   = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")

            self.metrics_[name] = {
                **val_metrics,
                "cv_f1_mean": round(float(cv_scores.mean()), 4),
                "cv_f1_std":  round(float(cv_scores.std()),  4),
            }

            m = self.metrics_[name]
            print(
                f"  val → acc={m['accuracy']}  f1={m['f1']}  "
                f"auc={m['roc_auc']}  |  cv_f1={m['cv_f1_mean']}±{m['cv_f1_std']}"
            )

        # pick best by val F1
        self.best_name_  = max(self.metrics_, key=lambda n: self.metrics_[n]["f1"])
        self.best_model_ = self.trained_[self.best_name_]
        print(f"\n[trainer] Best model: {self.best_name_}  "
              f"(val F1 = {self.metrics_[self.best_name_]['f1']})")

        return self.metrics_

    # ── test evaluation ───────────────────────────────────────────────────────

    def evaluate_best(
        self,
        X_test: sp.spmatrix,
        y_test: np.ndarray,
        classes: list[str] | None = None,
    ) -> dict:
        """Evaluate the best model on held-out test data."""
        if self.best_model_ is None:
            raise RuntimeError("Call train_all() first.")

        metrics = evaluate(self.best_model_, X_test, y_test, split_name="test")
        self.metrics_[self.best_name_]["test"] = metrics

        print(f"\n[trainer] Test results for {self.best_name_}:")
        print(f"  accuracy  = {metrics['accuracy']}")
        print(f"  precision = {metrics['precision']}")
        print(f"  recall    = {metrics['recall']}")
        print(f"  f1        = {metrics['f1']}")
        print(f"  roc_auc   = {metrics['roc_auc']}")

        if classes:
            y_pred = self.best_model_.predict(X_test)
            print("\n" + classification_report(y_test, y_pred, target_names=classes))

        return metrics

    # ── persistence ───────────────────────────────────────────────────────────

    def save_best(self, name: str | None = None):
        path = MODELS_DIR / (name or f"{self.best_name_}.joblib")
        joblib.dump(self.best_model_, path)
        print(f"[trainer] Best model saved → {path}")

        # Also save a generic 'best_model.joblib' for the API
        generic = MODELS_DIR / "best_model.joblib"
        joblib.dump(self.best_model_, generic)

        # Save metrics summary
        metrics_path = MODELS_DIR / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_, f, indent=2)
        print(f"[trainer] Metrics saved   → {metrics_path}")

    def save_all(self):
        for name, model in self.trained_.items():
            joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"[trainer] All models saved to {MODELS_DIR}")

    @staticmethod
    def load_model(name: str = "best_model") -> Any:
        path = MODELS_DIR / f"{name}.joblib"
        return joblib.load(path)

    # ── summary table ─────────────────────────────────────────────────────────

    def summary_df(self) -> pd.DataFrame:
        rows = []
        for name, m in self.metrics_.items():
            rows.append({
                "model":      name,
                "val_acc":    m.get("accuracy"),
                "val_f1":     m.get("f1"),
                "val_auc":    m.get("roc_auc"),
                "cv_f1_mean": m.get("cv_f1_mean"),
                "cv_f1_std":  m.get("cv_f1_std"),
                "best":       name == self.best_name_,
            })
        return pd.DataFrame(rows).sort_values("val_f1", ascending=False)


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=600, n_features=200, random_state=42)
    X    = sp.csr_matrix(X - X.min())  # make non-negative for NB

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_v,  X_te,  y_v,  y_te  = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

    trainer = ModelTrainer()
    trainer.train_all(X_tr, y_tr, X_v, y_v, cv_folds=3)
    trainer.evaluate_best(X_te, y_te, classes=["real", "fake"])
    print(trainer.summary_df().to_string(index=False))
