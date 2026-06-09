"""
app.py
------
Enhanced Flask REST API for the Fake News Detection system.
Includes MongoDB persistence for predictions history and analytics.

Endpoints
---------
GET  /health              → status check
POST /predict             → classify one article (saved to MongoDB)
POST /predict/batch       → classify many articles
GET  /metrics             → training metrics summary
GET  /models              → list available models
GET  /history             → paginated prediction history from MongoDB
GET  /analytics           → aggregated stats from MongoDB
DELETE /history/<id>      → delete a prediction record
GET  /news/latest         → fetch & analyze latest headlines
GET  /news/search?q=      → search & analyze news by keyword
"""

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from bson.json_util import dumps
import os
import requests as req

# ── app setup ──────────────────────────────────────────────────────────────────
_predictor = None
MODELS_DIR = Path(__file__).resolve().parent / "models"

app = Flask(__name__)
CORS(app)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    response = app.make_default_options_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    return response

# ── MongoDB setup ──────────────────────────────────────────────────────────────
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/fakenews?retryWrites=true&w=majority"
)

NEWS_API_KEY    = os.environ.get("NEWS_API_KEY", "")
GOOGLE_API_KEY  = os.environ.get("GOOGLE_API_KEY", "")

_db = None


# ── fact-check helpers ────────────────────────────────────────────────────────

def run_fact_checks(title: str) -> dict:
    """
    Run both Google Fact Check API (Option 1) and NewsAPI source verification
    (Option 2) in parallel and return a combined verification dict.
    """
    import concurrent.futures

    def google_fact_check(query):
        if not GOOGLE_API_KEY or not query:
            return []
        try:
            url = (
                "https://factchecktools.googleapis.com/v1alpha1/claims:search"
                f"?query={req.utils.quote(query)}&pageSize=5&key={GOOGLE_API_KEY}"
            )
            resp = req.get(url, timeout=8)
            claims = resp.json().get("claims", [])
            results = []
            for c in claims[:3]:
                review = (c.get("claimReview") or [{}])[0]
                results.append({
                    "claim":   c.get("text", ""),
                    "rating":  review.get("textualRating", ""),
                    "source":  review.get("publisher", {}).get("name", ""),
                    "url":     review.get("url", ""),
                })
            return results
        except Exception:
            return []

    def newsapi_source_check(query):
        if not NEWS_API_KEY or not query:
            return {"found_in_sources": 0, "source_verification": "API key not configured", "top_sources": []}
        try:
            url = (
                "https://newsapi.org/v2/everything"
                f"?q={req.utils.quote(query)}&pageSize=10&sortBy=relevancy"
                f"&language=en&apiKey={NEWS_API_KEY}"
            )
            resp     = req.get(url, timeout=8)
            articles = resp.json().get("articles", [])
            # filter out removed articles
            articles = [a for a in articles if a.get("title") and a["title"] != "[Removed]"]
            count    = len(articles)

            if count >= 5:
                verdict = "Widely reported — found in multiple credible sources"
            elif count >= 2:
                verdict = "Found in a few sources — limited coverage"
            elif count == 1:
                verdict = "Found in only one source — verify independently"
            else:
                verdict = "Not found in news sources — could not verify"

            top_sources = []
            seen = set()
            for a in articles[:5]:
                src_name = (a.get("source") or {}).get("name", "")
                if src_name and src_name not in seen:
                    seen.add(src_name)
                    top_sources.append({
                        "name":      src_name,
                        "url":       a.get("url", ""),
                        "published": a.get("publishedAt", ""),
                    })

            return {
                "found_in_sources":    count,
                "source_verification": verdict,
                "top_sources":         top_sources,
            }
        except Exception as e:
            return {"found_in_sources": 0, "source_verification": f"Check failed: {e}", "top_sources": []}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fc_future  = ex.submit(google_fact_check, title)
        src_future = ex.submit(newsapi_source_check, title)
        fact_checks    = fc_future.result()
        source_check   = src_future.result()

    return {
        "fact_checks":   fact_checks,
        "source_check":  source_check,
    }


def compute_combined_verdict(fact_checks, source_check) -> dict:
    """
    Combines three signals into a final fake/real score.
    Weights: ML=20%, NewsAPI source count=40%, Google Fact Check=40%
    Returns fake_score (0-100), real_score (0-100), final_label, final_verdict.
    """

    # ── Signal 1: NewsAPI source count (50% weight) ─────────────────────
    src_count = source_check.get("found_in_sources", 0) if source_check else 0
    # More sources = more likely real; scale 0-10+ sources to 0-1
    src_real = min(src_count / 8.0, 1.0)   # 8+ sources → fully real signal
    src_fake = 1 - src_real

    # ── Signal 3: Google Fact Check (40% weight) ──────────────────────────
    fc_fake = 0.0
    fc_real = 0.0
    fc_neutral = 0.0

    if fact_checks:
        for fc in fact_checks:
            rating = (fc.get("rating") or "").lower()
            if any(w in rating for w in ["false", "fake", "incorrect", "mislead", "fabricat", "pants on fire"]):
                fc_fake += 1
            elif any(w in rating for w in ["true", "correct", "accurate", "verified"]):
                fc_real += 1
            else:
                fc_neutral += 1

        total_fc = fc_fake + fc_real + fc_neutral or 1
        fc_fake_score = fc_fake / total_fc
        fc_real_score = fc_real / total_fc
        # neutral claims split evenly
        fc_fake_score += (fc_neutral / total_fc) * 0.5
        fc_real_score += (fc_neutral / total_fc) * 0.5
    else:
        # No fact-check data — treat as neutral, split 50/50
        fc_fake_score = 0.5
        fc_real_score = 0.5

    # ── Weighted combination ──────────────────────────────────────────────
    W_SRC = 0.50
    W_FC  = 0.50

    final_fake = round((src_fake * W_SRC + fc_fake_score * W_FC) * 100, 1)
    final_real = round((src_real * W_SRC + fc_real_score * W_FC) * 100, 1)

    # normalise to 100
    total = final_fake + final_real
    if total > 0:
        final_fake = round(final_fake / total * 100, 1)
        final_real = round(100 - final_fake, 1)

    final_label = "fake" if final_fake >= 50 else "real"

    if final_fake >= 80:
        verdict = "Almost certainly fake"
    elif final_fake >= 65:
        verdict = "Very likely fake"
    elif final_fake >= 55:
        verdict = "Likely fake"
    elif final_fake >= 45:
        verdict = "Uncertain — manual review recommended"
    elif final_fake >= 35:
        verdict = "Likely real"
    elif final_fake >= 20:
        verdict = "Very likely real"
    else:
        verdict = "Almost certainly real"

    return {
        "combined_fake_pct":  final_fake,
        "combined_real_pct":  final_real,
        "combined_label":     final_label,
        "combined_verdict":   verdict,
        "signal_weights": {
            "news_sources": {"weight": "50%", "fake_score": round(src_fake * 100, 1),  "real_score": round(src_real * 100, 1)},
            "fact_check":   {"weight": "50%", "fake_score": round(fc_fake_score * 100, 1), "real_score": round(fc_real_score * 100, 1)},
        }
    }


def get_db():
    global _db
    if _db is None:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _db = client["fakenews"]
    return _db


def get_predictor():
    global _predictor
    if _predictor is None:
        from src.predictor import FakeNewsPredictor
        _predictor = FakeNewsPredictor.load()
    return _predictor


def serialize_doc(doc):
    """Convert MongoDB doc to JSON-serializable dict."""
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    return doc


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    model_ready = (MODELS_DIR / "best_model.joblib").exists()
    db_status = "disconnected"
    try:
        get_db().command("ping")
        db_status = "connected"
    except Exception:
        pass
    return jsonify({
        "status": "ok",
        "model_ready": model_ready,
        "database": db_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


# ── /predict ──────────────────────────────────────────────────────────────────

@app.post("/predict")
def predict():
    try:
        body    = request.get_json(force=True) or {}
        title   = str(body.get("title", "")).strip()
        text    = str(body.get("text",  "")).strip()
        explain = bool(body.get("explain", False))
        source  = str(body.get("source", "")).strip()

        if not title and not text:
            return jsonify({"error": "Provide at least one of 'title' or 'text'."}), 400

        predictor = get_predictor()
        result    = predictor.predict(title=title, text=text)

        if explain:
            result["top_features"] = predictor.top_features(title=title, text=text)

        # ── live verification (Option 1 + Option 2 combined) ──────────────
        if title:
            verification = run_fact_checks(title)
            result["fact_checks"]  = verification["fact_checks"]
            result["source_check"] = verification["source_check"]
        else:
            result["fact_checks"]  = []
            result["source_check"] = {"found_in_sources": 0,
                                      "source_verification": "No title provided for verification",
                                      "top_sources": []}

        # ── combined verdict (ML 20% + NewsAPI 40% + FactCheck 40%) ───────
        combined = compute_combined_verdict(
            fact_checks  = result["fact_checks"],
            source_check = result["source_check"],
        )
        result.update(combined)

        record = {
            "title":         title,
            "text":          text[:2000],
            "source":        source,
            "label":         result["label"],
            "confidence":    result["confidence"],
            "verdict":       result["verdict"],
            "probabilities": result["probabilities"],
            "top_features":  result.get("top_features", []),
            "fact_checks":         result.get("fact_checks", []),
            "source_check":        result.get("source_check", {}),
            "combined_fake_pct":   result.get("combined_fake_pct"),
            "combined_real_pct":   result.get("combined_real_pct"),
            "combined_label":      result.get("combined_label"),
            "combined_verdict":    result.get("combined_verdict"),
            "created_at":          datetime.now(timezone.utc),
        }
        try:
            inserted = get_db()["predictions"].insert_one(record)
            result["id"] = str(inserted.inserted_id)
        except Exception as db_err:
            result["db_warning"] = f"Prediction succeeded but not saved: {db_err}"

        return jsonify(result)

    except FileNotFoundError:
        return jsonify({"error": "Model not found. Run train.py first."}), 503
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /predict/batch ────────────────────────────────────────────────────────────

@app.post("/predict/batch")
def predict_batch():
    try:
        import pandas as pd
        body = request.get_json(force=True) or []
        if not isinstance(body, list):
            return jsonify({"error": "Body must be a JSON array."}), 400

        df        = pd.DataFrame(body)
        predictor = get_predictor()
        out       = predictor.predict_batch(df)
        records   = out.to_dict(orient="records")

        try:
            now  = datetime.now(timezone.utc)
            docs = [{**r, "created_at": now, "batch": True} for r in records]
            get_db()["predictions"].insert_many(docs)
        except Exception:
            pass

        return jsonify(records)

    except FileNotFoundError:
        return jsonify({"error": "Model not found. Run train.py first."}), 503
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /metrics ──────────────────────────────────────────────────────────────────

@app.get("/metrics")
def metrics():
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        return jsonify({"error": "metrics.json not found."}), 404
    with open(metrics_path) as f:
        return jsonify(json.load(f))


# ── /models ───────────────────────────────────────────────────────────────────

@app.get("/models")
def list_models():
    if not MODELS_DIR.exists():
        return jsonify([])
    files = [p.stem for p in MODELS_DIR.glob("*.joblib")]
    return jsonify(files)


# ── /history ──────────────────────────────────────────────────────────────────

@app.get("/history")
def history():
    try:
        page     = max(1, int(request.args.get("page", 1)))
        per_page = min(50, int(request.args.get("per_page", 10)))
        label    = request.args.get("label")
        skip     = (page - 1) * per_page

        query = {}
        if label in ("fake", "real"):
            query["label"] = label

        col   = get_db()["predictions"]
        total = col.count_documents(query)
        docs  = list(col.find(query).sort("created_at", DESCENDING).skip(skip).limit(per_page))

        return jsonify({
            "total":    total,
            "page":     page,
            "per_page": per_page,
            "pages":    (total + per_page - 1) // per_page,
            "results":  [serialize_doc(d) for d in docs],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /analytics ────────────────────────────────────────────────────────────────

@app.get("/analytics")
def analytics():
    try:
        col   = get_db()["predictions"]
        total = col.count_documents({})

        if total == 0:
            return jsonify({"total": 0, "fake_count": 0, "real_count": 0,
                            "fake_pct": 0, "real_pct": 0, "avg_confidence": 0,
                            "recent_trend": []})

        fake_count = col.count_documents({"label": "fake"})
        real_count = col.count_documents({"label": "real"})

        agg      = list(col.aggregate([{"$group": {"_id": None, "avg_conf": {"$avg": "$confidence"}}}]))
        avg_conf = round(agg[0]["avg_conf"], 4) if agg else 0

        from datetime import timedelta
        trend = []
        for i in range(6, -1, -1):
            day_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=i)
            day_end   = day_start + timedelta(days=1)
            day_total = col.count_documents({"created_at": {"$gte": day_start, "$lt": day_end}})
            day_fake  = col.count_documents({"created_at": {"$gte": day_start, "$lt": day_end}, "label": "fake"})
            trend.append({
                "date":  day_start.strftime("%b %d"),
                "total": day_total,
                "fake":  day_fake,
                "real":  day_total - day_fake,
            })

        return jsonify({
            "total":          total,
            "fake_count":     fake_count,
            "real_count":     real_count,
            "fake_pct":       round(fake_count / total * 100, 1),
            "real_pct":       round(real_count / total * 100, 1),
            "avg_confidence": avg_conf,
            "recent_trend":   trend,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── DELETE /history/<id> ──────────────────────────────────────────────────────

@app.delete("/history/<record_id>")
def delete_history(record_id):
    try:
        result = get_db()["predictions"].delete_one({"_id": ObjectId(record_id)})
        if result.deleted_count == 0:
            return jsonify({"error": "Record not found."}), 404
        return jsonify({"deleted": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /news/latest ──────────────────────────────────────────────────────────────

@app.get("/news/latest")
def latest_news():
    """Fetch latest headlines and auto-analyze them."""
    if not NEWS_API_KEY:
        return jsonify({"error": "NEWS_API_KEY not configured."}), 503
    try:
        url      = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=20&apiKey={NEWS_API_KEY}"
        response = req.get(url, timeout=10)
        articles = response.json().get("articles", [])

        predictor = get_predictor()
        results   = []

        for article in articles:
            title       = article.get("title", "") or ""
            description = article.get("description", "") or ""
            source      = article.get("url", "")

            if not title or title == "[Removed]":
                continue

            prediction = predictor.predict(title=title, text=description)
            results.append({
                "title":      title,
                "source":     source,
                "image":      article.get("urlToImage", ""),
                "published":  article.get("publishedAt", ""),
                "label":      prediction["label"],
                "confidence": prediction["confidence"],
                "verdict":    prediction["verdict"],
            })

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /news/search ──────────────────────────────────────────────────────────────

@app.get("/news/search")
def search_news():
    """Search news by keyword and analyze."""
    if not NEWS_API_KEY:
        return jsonify({"error": "NEWS_API_KEY not configured."}), 503

    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Provide a search query ?q="}), 400

    try:
        url      = f"https://newsapi.org/v2/everything?q={query}&pageSize=15&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
        response = req.get(url, timeout=10)
        articles = response.json().get("articles", [])

        predictor = get_predictor()
        results   = []

        for article in articles:
            title  = article.get("title", "") or ""
            text   = article.get("description", "") or ""
            source = article.get("url", "")

            if not title or title == "[Removed]":
                continue

            prediction = predictor.predict(title=title, text=text)
            results.append({
                "title":      title,
                "source":     source,
                "image":      article.get("urlToImage", ""),
                "published":  article.get("publishedAt", ""),
                "label":      prediction["label"],
                "confidence": prediction["confidence"],
                "verdict":    prediction["verdict"],
            })

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /verify ───────────────────────────────────────────────────────────────────

@app.get("/verify")
def verify():
    """Standalone fact-check + source verification for a query string."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Provide ?q= parameter"}), 400
    try:
        return jsonify(run_fact_checks(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Fake News Detection API …")
    app.run(host="0.0.0.0", port=5000, debug=True)