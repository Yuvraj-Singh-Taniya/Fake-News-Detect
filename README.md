# TruthLens — Fake News Detector

Full-stack integration of the fake news detection ML model with:
- **Flask** backend (REST API)
- **MongoDB Atlas** (cloud database for prediction history & analytics)
- **React + Vite** frontend (clean, dark-mode UI)

---

## Project Structure

```
fakenews-fullstack/
├── backend/
│   ├── app.py                  ← Enhanced Flask API with MongoDB
│   ├── requirements.txt
│   ├── .env.example            ← Copy to .env and fill in MongoDB URI
│   ├── models/                 ← Your trained .joblib model files
│   └── src/                    ← predictor.py, preprocessor.py, etc.
│
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        └── App.jsx             ← Full UI (Detect, History, Analytics tabs)
```

---

## 1. MongoDB Atlas Setup

1. Go to [https://cloud.mongodb.com](https://cloud.mongodb.com) and create a free cluster
2. Create a database user (Database Access → Add New User)
3. Allow your IP (Network Access → Add IP Address → `0.0.0.0/0` for dev)
4. Get your connection string (Connect → Drivers → Python)
5. It looks like: `mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/`

---

## 2. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure MongoDB
cp .env.example .env
# Edit .env and set your MONGO_URI:
# MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/fakenews?retryWrites=true&w=majority

# Run the server
python app.py
```

The API will be live at **http://localhost:5000**

### API Endpoints

| Method | Endpoint             | Description                              |
|--------|----------------------|------------------------------------------|
| GET    | `/health`            | Status of API, model, and database       |
| POST   | `/predict`           | Classify one article (saved to MongoDB)  |
| POST   | `/predict/batch`     | Classify an array of articles            |
| GET    | `/metrics`           | ML model performance metrics             |
| GET    | `/models`            | List available model files               |
| GET    | `/history`           | Paginated prediction history             |
| GET    | `/analytics`         | Aggregated stats + 7-day trend           |
| DELETE | `/history/<id>`      | Delete a prediction record               |

#### POST /predict — Request Body
```json
{
  "title": "Vaccines cause autism, experts say",
  "text": "Optional full article body...",
  "source": "https://example.com/article",
  "explain": true
}
```

#### GET /history — Query Params
- `page` (default 1)
- `per_page` (default 10, max 50)
- `label` — filter by `fake` or `real`

---

## 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

The app will be live at **http://localhost:3000**

### Features
- **Detect tab** — paste a headline + body, get prediction with confidence gauge and word-level explanation
- **History tab** — paginated list of all past predictions stored in MongoDB, with expand/delete
- **Analytics tab** — total counts, fake/real ratio, 7-day trend chart, model performance metrics

---

## 4. Production Deployment

### Backend (e.g., Railway / Render / EC2)
```bash
pip install gunicorn
gunicorn app:app --bind 0.0.0.0:5000 --workers 2
```
Set `MONGO_URI` as an environment variable in your host's dashboard.

### Frontend (e.g., Vercel / Netlify)
```bash
npm run build
# Deploy the dist/ folder
```
Update the `API` constant in `src/App.jsx` from `http://localhost:5000` to your deployed backend URL.

---

## Environment Variables

| Variable   | Description                          | Example                                           |
|------------|--------------------------------------|---------------------------------------------------|
| `MONGO_URI` | MongoDB Atlas connection string     | `mongodb+srv://user:pass@cluster.mongodb.net/...` |
| `FLASK_ENV` | Flask environment                   | `development` or `production`                     |
| `FLASK_PORT`| Port to run on (default 5000)       | `5000`                                            |
