"""
preprocessor.py
---------------
Text cleaning and feature extraction for fake news detection.

Steps
-----
1. Combine title + body text
2. Lowercase, strip HTML/URLs/punctuation
3. Tokenise → remove stopwords → lemmatise  (NLTK)
4. Vectorise with TF-IDF (fit on train, transform all splits)
"""

import re
import string
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ── paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── NLTK bootstrap ────────────────────────────────────────────────────────────
def _ensure_nltk():
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            if pkg == "punkt":
                nltk.data.find("tokenizers/punkt")
            elif pkg == "punkt_tab":
                nltk.data.find("tokenizers/punkt_tab")
            elif pkg == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif pkg == "wordnet":
                nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk()

_STOP_WORDS  = set(stopwords.words("english"))
_LEMMATIZER  = WordNetLemmatizer()
_URL_RE      = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE     = re.compile(r"<[^>]+>")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ── text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for a single string.
    Returns a space-joined string of lemmatised tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = _HTML_RE.sub(" ", text)           # strip HTML tags
    text = _URL_RE.sub(" ", text)            # strip URLs
    text = text.lower()                      # lowercase
    text = text.translate(_PUNCT_TABLE)      # remove punctuation
    text = re.sub(r"\d+", " ", text)         # remove digits
    text = re.sub(r"\s+", " ", text).strip() # normalise whitespace

    tokens = word_tokenize(text)
    tokens = [
        _LEMMATIZER.lemmatize(t)
        for t in tokens
        if t not in _STOP_WORDS and len(t) > 2
    ]
    return " ".join(tokens)


def combine_and_clean(df: pd.DataFrame) -> pd.Series:
    """
    Merge title + text columns, then clean.
    Works whether the DataFrame has 'title', 'text', or both.
    """
    title = df.get("title", pd.Series([""] * len(df))).fillna("")
    body  = df.get("text",  pd.Series([""] * len(df))).fillna("")
    combined = title + " " + body
    return combined.map(clean_text)


# ── label encoding ────────────────────────────────────────────────────────────

def encode_labels(series: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode 'real'/'fake' → 0/1.
    Returns (encoded_array, fitted_LabelEncoder).
    """
    le = LabelEncoder()
    encoded = le.fit_transform(series)
    return encoded, le


# ── TF-IDF vectoriser ─────────────────────────────────────────────────────────

def build_tfidf(
    max_features: int  = 50_000,
    ngram_range:  tuple = (1, 2),
    min_df:       int  = 2,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Return a configured (unfitted) TfidfVectorizer."""
    return TfidfVectorizer(
        max_features = max_features,
        ngram_range  = ngram_range,
        min_df       = min_df,
        sublinear_tf = sublinear_tf,
        analyzer     = "word",
        token_pattern= r"\b[a-z]{3,}\b",
    )


# ── high-level API ────────────────────────────────────────────────────────────

class TextPreprocessor:
    """
    Stateful wrapper: fits a TF-IDF vectoriser on training data
    and transforms any split on demand.

    Usage
    -----
    >>> pp = TextPreprocessor()
    >>> X_train = pp.fit_transform(train_df)
    >>> X_val   = pp.transform(val_df)
    >>> X_test  = pp.transform(test_df)
    >>> y_train, y_val, y_test = pp.encode_splits(train_df, val_df, test_df)
    """

    def __init__(self, max_features: int = 50_000):
        self.vectorizer  = build_tfidf(max_features=max_features)
        self.label_enc   = LabelEncoder()
        self._fitted     = False

    # ── fit / transform ───────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame):
        cleaned = combine_and_clean(df)
        X = self.vectorizer.fit_transform(cleaned)
        self._fitted = True
        return X

    def transform(self, df: pd.DataFrame):
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on training data first.")
        cleaned = combine_and_clean(df)
        return self.vectorizer.transform(cleaned)

    # ── labels ────────────────────────────────────────────────────────────────

    def fit_labels(self, series: pd.Series) -> np.ndarray:
        return self.label_enc.fit_transform(series)

    def transform_labels(self, series: pd.Series) -> np.ndarray:
        return self.label_enc.transform(series)

    def encode_splits(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        test_df:  pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_train = self.fit_labels(train_df["label"])
        y_val   = self.transform_labels(val_df["label"])
        y_test  = self.transform_labels(test_df["label"])
        return y_train, y_val, y_test

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path | None = None):
        path = Path(path or MODELS_DIR / "preprocessor.joblib")
        joblib.dump(self, path)
        print(f"[preprocessor] Saved → {path}")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "TextPreprocessor":
        path = Path(path or MODELS_DIR / "preprocessor.joblib")
        obj  = joblib.load(path)
        print(f"[preprocessor] Loaded ← {path}")
        return obj

    # ── convenience ───────────────────────────────────────────────────────────

    @property
    def classes_(self) -> list[str]:
        return list(self.label_enc.classes_)

    @property
    def vocab_size(self) -> int:
        return len(self.vectorizer.vocabulary_) if self._fitted else 0


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = pd.DataFrame({
        "title": ["Scientists confirm vaccine safety in new study",
                  "SHOCKING: Government hiding secret about vaccines!!!"],
        "text":  ["Peer-reviewed research published in The Lancet confirms ...",
                  "You won't believe what they found — share before deleted ..."],
        "label": ["real", "fake"],
    })

    pp = TextPreprocessor(max_features=500)
    X  = pp.fit_transform(sample)
    y  = pp.fit_labels(sample["label"])

    print("Vocab size :", pp.vocab_size)
    print("Matrix shape:", X.shape)
    print("Labels      :", y, "→", pp.classes_)
    print("Cleaned text sample:")
    print(combine_and_clean(sample).tolist())
