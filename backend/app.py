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

# ── app setup ──────────────────────────────────────────────────────────────────
_predictor = None
MODELS_DIR = Path(__file__).resolve().parent / "models"

app = Flask(__name__)
CORS(app, origins="*")

# ── MongoDB setup ──────────────────────────────────────────────────────────────
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/fakenews?retryWrites=true&w=majority"
)

_db = None

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
        source  = str(body.get("source", "")).strip()  # optional source URL

        if not title and not text:
            return jsonify({"error": "Provide at least one of 'title' or 'text'."}), 400

        predictor = get_predictor()
        result    = predictor.predict(title=title, text=text)

        if explain:
            result["top_features"] = predictor.top_features(title=title, text=text)

        # ── save to MongoDB ────────────────────────────────────────────────────
        record = {
            "title":       title,
            "text":        text[:2000],  # cap stored text to 2KB
            "source":      source,
            "label":       result["label"],
            "confidence":  result["confidence"],
            "verdict":     result["verdict"],
            "probabilities": result["probabilities"],
            "top_features": result.get("top_features", []),
            "created_at":  datetime.now(timezone.utc),
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

        # bulk insert
        try:
            now = datetime.now(timezone.utc)
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
    """Return paginated prediction history."""
    try:
        page     = max(1, int(request.args.get("page", 1)))
        per_page = min(50, int(request.args.get("per_page", 10)))
        label    = request.args.get("label")        # filter: fake|real
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
    """Return aggregated analytics from stored predictions."""
    try:
        col = get_db()["predictions"]
        total = col.count_documents({})

        if total == 0:
            return jsonify({"total": 0, "fake_count": 0, "real_count": 0,
                            "fake_pct": 0, "real_pct": 0, "avg_confidence": 0,
                            "recent_trend": []})

        fake_count = col.count_documents({"label": "fake"})
        real_count = col.count_documents({"label": "real"})

        # average confidence
        agg = list(col.aggregate([
            {"$group": {"_id": None, "avg_conf": {"$avg": "$confidence"}}}
        ]))
        avg_conf = round(agg[0]["avg_conf"], 4) if agg else 0

        # last 7 days daily breakdown
        from datetime import timedelta
        trend = []
        for i in range(6, -1, -1):
            day_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=i)
            day_end = day_start + timedelta(days=1)
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


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Fake News Detection API …")
    app.run(host="0.0.0.0", port=5000, debug=True)
