# WaferVision API Contract for React Week 3

Base URL during local dev:

```text
http://localhost:8000/api/v1
```

## 1. `GET /health`

Use this to show an API/model status badge.

```json
{
  "status": "ok",
  "app": "WaferVision API",
  "environment": "local",
  "model_loaded": true,
  "database": "sqlite:///./data/runtime/wafervision.db"
}
```

## 2. `GET /model`

Use this for the model details panel.

```json
{
  "loaded": true,
  "model_version": "wafer_cnn_best",
  "checkpoint_path": "artifacts/checkpoints/wafer_cnn_best.pt",
  "input_size": 64,
  "class_names": ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "None"],
  "device": "cpu",
  "validation_macro_f1": 0.91
}
```

## 3. `POST /predict`

Multipart upload.

Supported files:

```text
.csv, .npy, .png, .jpg, .jpeg, .bmp
```

Request:

```ts
const form = new FormData();
form.append("file", file);
form.append("note", "optional note");

await fetch(`${API_BASE}/predict`, {
  method: "POST",
  body: form,
});
```

Response:

```json
{
  "id": 1,
  "label": "Scratch",
  "confidence": 0.973,
  "top_k": [
    {"label": "Scratch", "probability": 0.973},
    {"label": "Loc", "probability": 0.018},
    {"label": "Random", "probability": 0.006}
  ],
  "inference_ms": 12.4,
  "created_at": "2026-06-17T00:00:00Z",
  "input": {
    "filename": "wafer.csv",
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

## 4. `POST /predict/array`

Quick dev endpoint. Send a 2D grid directly.

```json
{
  "filename": "inline-demo",
  "note": "optional",
  "wafer_map": [
    [0,0,0,0],
    [0,1,2,0],
    [0,1,1,0],
    [0,0,0,0]
  ]
}
```

Response is identical to `/predict`.

## 5. `GET /predictions`

Query params:

```text
limit: 1..200, default 50
offset: >=0, default 0
label: optional exact label filter
```

Response:

```json
{
  "total": 42,
  "limit": 50,
  "offset": 0,
  "items": [
    {
      "id": 42,
      "created_at": "2026-06-17T00:00:00Z",
      "filename": "wafer.csv",
      "input_kind": "upload",
      "predicted_label": "Edge-Ring",
      "confidence": 0.94,
      "inference_ms": 15.1,
      "rows": 52,
      "cols": 52,
      "model_version": "wafer_cnn_best"
    }
  ]
}
```

## 6. `GET /predictions/{id}`

Full record including `top_k`, input min/max, checkpoint path, and client note.

## 7. `GET /stats/summary`

Use this for dashboard cards and charts.

```json
{
  "total_predictions": 42,
  "average_confidence": 0.91,
  "label_counts": [
    {"label": "Edge-Ring", "count": 12},
    {"label": "Scratch", "count": 8}
  ],
  "latest": []
}
```


## 8. `POST /simulator/run`

Creates a synthetic wafer-lot simulation and returns all wafer maps plus dashboard summaries. The endpoint works with a loaded CNN/SVM model or a heuristic fallback.

Request:

```json
{
  "wafer_count": 48,
  "size": 64,
  "seed": 20260617,
  "lot_count": 4,
  "noise_level": 0.045,
  "defect_density_scale": 1.1,
  "mixed_pattern_rate": 0.12,
  "scenario_name": "edge-ring-excursion",
  "pattern_mix": {
    "Edge-Ring": 0.52,
    "Edge-Loc": 0.18,
    "Loc": 0.08
  },
  "persist": true
}
```

Response includes:

```text
session_id
created_at
params
wafers[]       # matrix + lot/tool/chamber metadata + prediction + risk/yield metrics
summary        # counts, tool risk, confusion, trend
model          # model metadata
```

## 9. `GET /simulator/sessions`

Returns saved simulator sessions from SQLite.

## 10. `GET /simulator/sessions/{session_id}`

Loads one saved simulator session, including wafer matrices and all dashboard-ready summaries.
