## Synthetic data generator

This project includes a script to generate anonymized user journeys for the fashion catalog.

### Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python .\src\generate_data.py
```

Outputs are written to `generated_data/` as CSV files matching the schema:
- `profiles.csv`
- `product_interactions.csv`
- `wishlist_items.csv`
- `cart_items.csv`
- `orders.csv`
- `order_items.csv`

Notes:
- 30 anonymized profiles are created with consistent personas (age, gender, style affinities).
- Journeys simulate realistic search → view → wishlist/cart → checkout behavior with cross-gender browsing allowed but not dominant.
- Timestamps span recent weeks for temporal variety.

---

## FastAPI service

This repo includes a FastAPI service to serve recommendations and basic metadata for a React frontend.

### Endpoints

- `GET /health`: service status.
- `POST /train`: loads data and trains lightweight models (BPR + TF-IDF profiles).
- `GET /methods`: list available recommender methods.
- `POST /recommend`:
  - body: `{ "user_id": "u_123", "method": "blend_bpr_tfidf", "top_k": 20 }`
  - response: `{ "user_id": "u_123", "method": "blend_bpr_tfidf", "items": ["item1", ...] }`
- `GET /items/{item_id}`: returns product metadata for a given item id.
- `GET /metrics`: returns latest experiment metrics if present.

### Local run

Install dependencies and start the server:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Train models (once per run) then request recommendations:

```bash
curl -X POST http://localhost:8000/train
curl -X POST http://localhost:8000/recommend -H 'Content-Type: application/json' \
  -d '{"user_id":"1","method":"blend_bpr_tfidf","top_k":20}'
```

### CORS

Default allowed origins: `http://localhost:3000`. Override with env `ALLOW_ORIGINS` (comma-separated).

### Docker

Build and run on Ubuntu VM:

```bash
docker build -t fashion-recsys:latest .
docker run --rm -p 8000:8000 \
  -e ALLOW_ORIGINS="http://localhost:3000" \
  fashion-recsys:latest
```

The container exposes `8000` and starts `uvicorn` pointing to `src.api.app:app`.
