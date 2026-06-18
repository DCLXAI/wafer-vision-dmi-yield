# WaferVision — Week 2 FastAPI Backend

Week 2 turns the Week 1 PyTorch checkpoint into a real model-serving backend:

```text
React upload UI
    ↓ multipart/form-data / JSON array
FastAPI backend
    ↓ decode CSV / NPY / PNG / JPG wafer map
PyTorch WaferCNN checkpoint
    ↓ softmax prediction
SQLite prediction history
    ↓ dashboard analytics API
React Recharts dashboard
```

## What is implemented

- FastAPI app with OpenAPI docs
- PyTorch model service that loads `wafer_cnn_best.pt` once at startup
- Upload prediction endpoint for `.csv`, `.npy`, `.png`, `.jpg`, `.jpeg`, `.bmp`
- JSON array prediction endpoint for quick frontend testing
- SQLite history table for every inference
- History list, detail, and summary statistics endpoints
- CORS ready for React on Vite or Next.js
- Test suite with dummy checkpoint
- Dockerfile and `.env.example`
- Synthetic sample wafer maps for API smoke tests

## Project structure

```text
src/
  wafer_vision/              # Week 1 training/inference code
  wafer_vision_api/          # Week 2 FastAPI backend
    app.py                   # FastAPI factory + lifespan
    settings.py              # env-based config
    database.py              # SQLite/SQLAlchemy setup
    db_models.py             # prediction_records table
    input_loader.py          # CSV/NPY/image upload decoder
    schemas.py               # API response/request models
    routes/
      health.py
      predict.py
      history.py
    services/
      model_service.py       # checkpoint loading + inference
      history_service.py     # SQLite persistence + stats
scripts/
  create_demo_checkpoint.py  # untrained checkpoint for wiring checks only
  create_sample_inputs.py    # synthetic CSV/NPY/PNG samples
  api_smoke_test.py          # local API smoke test through TestClient
tests/
  test_api.py
  test_input_loader.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

Expected result:

```text
11 passed
```

## Real model path

After Week 1 training, the backend expects:

```text
artifacts/checkpoints/wafer_cnn_best.pt
```

The checkpoint must contain:

```python
{
  "model_state_dict": ...,
  "class_names": [...],
  "input_size": 64,
  "config": {...},
  "val_metrics": {...}
}
```

That is already the format produced by the Week 1 trainer.

## Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Main settings:

```bash
WAFERVISION_CHECKPOINT_PATH=artifacts/checkpoints/wafer_cnn_best.pt
WAFERVISION_DATABASE_URL=sqlite:///./data/runtime/wafervision.db
WAFERVISION_DEVICE=auto
WAFERVISION_TOP_K=5
WAFERVISION_MAX_UPLOAD_MB=10
WAFERVISION_CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

## Start API

```bash
PYTHONPATH=src uvicorn wafer_vision_api.app:app --reload --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

If the checkpoint is missing, the API still boots in degraded mode. `/api/v1/health` will show `model_loaded: false`, and `/api/v1/predict` will return `503` until a checkpoint is available.

## Smoke test without a trained model

This creates an **untrained** checkpoint only to verify API wiring. Do not claim this as a trained model.

```bash
PYTHONPATH=src python scripts/create_demo_checkpoint.py \
  --output artifacts/checkpoints/wafer_cnn_demo.pt

PYTHONPATH=src WAFERVISION_CHECKPOINT_PATH=artifacts/checkpoints/wafer_cnn_demo.pt \
  python scripts/api_smoke_test.py
```

Or with make:

```bash
make demo-checkpoint
PYTHONPATH=src WAFERVISION_CHECKPOINT_PATH=artifacts/checkpoints/wafer_cnn_demo.pt make api-smoke
```

## Upload prediction

### CSV / NPY / PNG / JPG upload

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@samples/synthetic_edge_ring.csv" \
  -F "note=demo upload"
```

Example response:

```json
{
  "id": 1,
  "label": "Edge-Ring",
  "confidence": 0.94,
  "top_k": [
    {"label": "Edge-Ring", "probability": 0.94},
    {"label": "Edge-Loc", "probability": 0.04}
  ],
  "inference_ms": 18.2,
  "created_at": "2026-06-17T00:00:00Z",
  "input": {
    "filename": "synthetic_edge_ring.csv",
    "content_type": "text/csv",
    "input_kind": "upload",
    "rows": 52,
    "cols": 52,
    "min_value": 0.0,
    "max_value": 2.0
  },
  "model": {
    "loaded": true,
    "model_version": "wafer_cnn_best",
    "checkpoint_path": "artifacts/checkpoints/wafer_cnn_best.pt",
    "input_size": 64,
    "class_names": ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "None"],
    "device": "cpu",
    "validation_macro_f1": 0.91
  }
}
```

### Inline JSON array prediction

Useful for React development before file upload UI is ready:

```bash
curl -X POST http://localhost:8000/api/v1/predict/array \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "inline-test",
    "note": "frontend smoke test",
    "wafer_map": [[0,0,0,0],[0,1,2,0],[0,1,1,0],[0,0,0,0]]
  }'
```

## History and analytics endpoints

```bash
# List latest predictions
curl "http://localhost:8000/api/v1/predictions?limit=20&offset=0"

# Filter by label
curl "http://localhost:8000/api/v1/predictions?label=Scratch"

# One prediction detail
curl http://localhost:8000/api/v1/predictions/1

# Dashboard summary
curl http://localhost:8000/api/v1/stats/summary

# Model metadata
curl http://localhost:8000/api/v1/model
```

## SQLite schema

Table: `prediction_records`

```text
id
created_at
filename
content_type
input_kind
rows
cols
min_value
max_value
predicted_label
confidence
top_k_json
model_version
checkpoint_path
inference_ms
client_note
```

This is intentionally simple and perfect for the Week 3 React dashboard:

- prediction history table
- defect type distribution chart
- average confidence KPI
- latest upload result card
- model status badge

## React integration contract

Frontend upload call:

```ts
export async function predictWafer(file: File) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch("http://localhost:8000/api/v1/predict", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}
```

Dashboard calls:

```ts
const summary = await fetch("http://localhost:8000/api/v1/stats/summary").then(r => r.json());
const history = await fetch("http://localhost:8000/api/v1/predictions?limit=50").then(r => r.json());
const model = await fetch("http://localhost:8000/api/v1/model").then(r => r.json());
```

## Deploy notes

For Render/Fly/Railway:

1. Build command:

```bash
pip install -r requirements.txt
```

2. Start command:

```bash
uvicorn wafer_vision_api.app:app --host 0.0.0.0 --port $PORT
```

3. Add environment variables:

```bash
WAFERVISION_CHECKPOINT_PATH=artifacts/checkpoints/wafer_cnn_best.pt
WAFERVISION_DATABASE_URL=sqlite:///./data/runtime/wafervision.db
WAFERVISION_DEVICE=cpu
```

For production-grade deployment, move from SQLite to Postgres and store checkpoints in object storage, but SQLite is exactly right for a 1-month portfolio demo.
