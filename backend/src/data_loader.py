"""
data_loader.py
--------------
Downloads and prepares FakeNewsNet data for the fake news detection pipeline.

FakeNewsNet has two sources:
  - PolitiFact  (political news, human fact-checked)
  - GossipCop   (celebrity/entertainment news)

Each source has real/ and fake/ subdirectories on GitHub:
  https://github.com/KaiDMML/FakeNewsNet
"""

import os
import json
import time
import random
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# GitHub raw content base for FakeNewsNet
BASE_URL = (
    "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset"
)

SOURCES = ["politifact", "gossipcop"]
LABELS  = ["real", "fake"]

# ── helpers ──────────────────────────────────────────────────────────────────

def _fetch_json(url: str, retries: int = 3) -> dict | None:
    """GET a JSON file; return None on failure."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def _fetch_news_ids(source: str, label: str) -> list[str]:
    """
    Pull the list of news IDs from the dataset index CSV stored on GitHub.
    Falls back to an empty list on any network/parse error.
    """
    csv_url = f"{BASE_URL}/{source}_{label}_content.csv"
    try:
        df = pd.read_csv(csv_url)
        if "id" in df.columns:
            return df["id"].astype(str).tolist()
        return df.iloc[:, 0].astype(str).tolist()
    except Exception:
        return []


# ── main download ─────────────────────────────────────────────────────────────

def download_fakenewsnet(
    max_per_class: int = 500,
    sources: list[str] | None = None,
) -> pd.DataFrame:
    """
    Download articles from FakeNewsNet and return a combined DataFrame.

    Parameters
    ----------
    max_per_class : int
        Maximum number of articles to fetch per (source, label) combination.
    sources : list[str] | None
        Subset of ['politifact', 'gossipcop'].  Defaults to both.

    Returns
    -------
    pd.DataFrame  with columns: id, title, text, label, source
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / "fakenewsnet_raw.csv"

    if cache_path.exists():
        print(f"[data_loader] Using cached data at {cache_path}")
        return pd.read_csv(cache_path)

    sources = sources or SOURCES
    records: list[dict] = []

    for source in sources:
        for label in LABELS:
            print(f"\n[data_loader] Fetching {source}/{label} …")
            ids = _fetch_news_ids(source, label)
            if not ids:
                print(f"  ⚠  No IDs found for {source}/{label} — skipping.")
                continue

            random.shuffle(ids)
            fetched = 0

            for news_id in tqdm(ids, desc=f"{source}/{label}"):
                if fetched >= max_per_class:
                    break
                url = f"{BASE_URL}/{source}/{label}/{news_id}/news content.json"
                data = _fetch_json(url)
                if data is None:
                    continue

                title = data.get("title", "") or ""
                text  = data.get("text",  "") or ""
                if len(title) + len(text) < 30:
                    continue  # skip near-empty articles

                records.append({
                    "id":     news_id,
                    "title":  title.strip(),
                    "text":   text.strip(),
                    "label":  label,          # "real" | "fake"
                    "source": source,
                })
                fetched += 1
                time.sleep(0.05)  # be polite to GitHub CDN

    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    print(f"\n[data_loader] Saved {len(df)} articles → {cache_path}")
    return df


# ── fallback: generate synthetic data ────────────────────────────────────────

def generate_synthetic_data(n_per_class: int = 300) -> pd.DataFrame:
    """
    Generate a small synthetic dataset so the pipeline can run offline.
    Useful for quick smoke-tests without hitting GitHub.
    """
    real_templates = [
        "Scientists confirm that {topic} improves health outcomes in new study.",
        "Government officials announce new policy regarding {topic}.",
        "Researchers at MIT publish findings on {topic} in peer-reviewed journal.",
        "Stock markets respond to {topic} news as investors weigh implications.",
        "New legislation addresses concerns about {topic} following public debate.",
    ]
    fake_templates = [
        "SHOCKING: {topic} secretly causes cancer — doctors don't want you to know!",
        "BREAKING: Government hiding truth about {topic} from the public!!!",
        "{topic} EXPOSED as massive hoax — share before they delete this!",
        "You won't believe what they found in {topic} — mainstream media silent.",
        "Elite globalists use {topic} to control the population — insider reveals all.",
    ]
    topics = [
        "climate change", "vaccines", "5G towers", "artificial intelligence",
        "cryptocurrency", "elections", "solar energy", "water fluoridation",
        "gene editing", "social media", "electric vehicles", "space exploration",
    ]

    rng = random.Random(42)
    records = []

    for label, templates in [("real", real_templates), ("fake", fake_templates)]:
        for i in range(n_per_class):
            topic    = rng.choice(topics)
            template = rng.choice(templates)
            title    = template.format(topic=topic)
            # Pad with extra filler so text ≠ title
            filler   = " ".join(rng.choices(title.split(), k=rng.randint(30, 80)))
            records.append({
                "id":     f"synth_{label}_{i:04d}",
                "title":  title,
                "text":   f"{title} {filler}",
                "label":  label,
                "source": "synthetic",
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "synthetic.csv", index=False)
    return df


# ── split ────────────────────────────────────────────────────────────────────

def train_val_test_split(
    df: pd.DataFrame,
    train: float = 0.70,
    val:   float = 0.15,
    seed:  int   = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train / validation / test."""
    from sklearn.model_selection import train_test_split

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    X_train, X_tmp = train_test_split(
        df, test_size=1 - train, stratify=df["label"], random_state=seed
    )
    rel_val = val / (1 - train)
    X_val, X_test = train_test_split(
        X_tmp, test_size=1 - rel_val, stratify=X_tmp["label"], random_state=seed
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    X_val.to_csv(PROCESSED_DIR  / "val.csv",   index=False)
    X_test.to_csv(PROCESSED_DIR / "test.csv",  index=False)

    print(f"[split] train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    return X_train, X_val, X_test


if __name__ == "__main__":
    # Quick smoke-test with synthetic data
    df = generate_synthetic_data(n_per_class=100)
    print(df["label"].value_counts())
    train, val, test = train_val_test_split(df)
    print(train.head(3))
