# WaferVision — Week 1 AI Model

**WaferVision** is a portfolio-grade semiconductor wafer defect classification project.
Week 1 focuses on the AI model: preprocessing WM-811K wafer maps, training a CNN baseline, evaluating with serious metrics, and saving a checkpoint ready for Week 2 FastAPI serving.

---

## 1. What this baseline does

```text
WM-811K LSWMD.pkl
      ↓
clean labels + resize wafer maps to 64x64
      ↓
PyTorch Dataset / DataLoader
      ↓
Compact CNN classifier
      ↓
accuracy + macro F1 + classification report + confusion matrix
      ↓
artifacts/checkpoints/wafer_cnn_best.pt
```

Default classes:

```text
Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, None
```

If you want defect-only classification, set this in `configs/train.yaml`:

```yaml
include_none: false
```

---

## 2. Dataset setup

Download the WM-811K public wafer-map benchmark:

```text
https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
```

Then place the pickle file here:

```text
data/raw/LSWMD.pkl
```

Expected structure:

```text
wafer-vision-week1/
  data/
    raw/
      LSWMD.pkl
```

Important: the dataset file is intentionally not included in this repo because it is large.

---

## 3. Local setup

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

or:

```bash
make test
```

---

## 4. Fast smoke training

Use this first to verify your environment before full training:

```bash
PYTHONPATH=src python -m wafer_vision.train \
  --config configs/train.yaml \
  --epochs 2 \
  --max-samples 5000
```

or:

```bash
make train-smoke
```

---

## 5. Full training

```bash
PYTHONPATH=src python -m wafer_vision.train --config configs/train.yaml
```

or:

```bash
make train
```

Outputs:

```text
artifacts/checkpoints/wafer_cnn_best.pt
artifacts/reports/metrics.json
artifacts/reports/confusion_matrix.csv
```

The checkpoint includes:

- model weights
- class names
- input size
- config
- validation metrics
- class counts

This makes it easy to load from FastAPI in Week 2.

---

## 6. Evaluate checkpoint

```bash
PYTHONPATH=src python -m wafer_vision.evaluate \
  --config configs/train.yaml \
  --checkpoint artifacts/checkpoints/wafer_cnn_best.pt
```

---

## 7. Predict one wafer map

Supported input formats: `.npy`, `.pkl`, `.csv`, `.png`, `.jpg`.

```bash
PYTHONPATH=src python -m wafer_vision.predict \
  --checkpoint artifacts/checkpoints/wafer_cnn_best.pt \
  --input path/to/wafer_map.npy
```

Example output:

```json
{
  "prediction": "Edge-Ring",
  "confidence": 0.9821,
  "top_k": [
    {"label": "Edge-Ring", "probability": 0.9821},
    {"label": "Edge-Loc", "probability": 0.0113}
  ]
}
```

---

## 8. Visualize sample wafer maps

```bash
PYTHONPATH=src python scripts/visualize_samples.py \
  --data-path data/raw/LSWMD.pkl \
  --output artifacts/reports/sample_wafer_maps.png
```

---

## 9. Portfolio explanation

Use this wording in your README / portfolio page:

> I built a semiconductor wafer defect classification system using the WM-811K industrial wafer map dataset. The model preprocesses variable-size wafer bin maps, trains a CNN classifier across major failure patterns, and reports accuracy, macro F1, per-class metrics, and confusion matrix. The trained PyTorch checkpoint is designed to be served through a FastAPI backend and visualized in a React dashboard.

---

## 10. Next improvements

Week 1 baseline is enough to prove ML + domain understanding. For a stronger final portfolio, add:

1. **Grad-CAM / occlusion heatmap** to explain where the model looked.
2. **Unknown defect detection** by confidence thresholding or embedding distance.
3. **Imbalance handling** with oversampling or focal loss.
4. **EfficientNet-style model** after the CNN baseline works.
5. **FastAPI `/predict` endpoint** loading `wafer_cnn_best.pt`.

---

# Week 2 — FastAPI Backend

The backend implementation lives in `src/wafer_vision_api` and serves the Week 1 checkpoint through FastAPI.

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest -q
```

Create sample inputs:

```bash
PYTHONPATH=src python scripts/create_sample_inputs.py
```

Run with the real Week 1 model:

```bash
PYTHONPATH=src uvicorn wafer_vision_api.app:app --reload --host 0.0.0.0 --port 8000
```

Run an API smoke test with an untrained demo checkpoint:

```bash
PYTHONPATH=src python scripts/create_demo_checkpoint.py --output artifacts/checkpoints/wafer_cnn_demo.pt
PYTHONPATH=src WAFERVISION_CHECKPOINT_PATH=artifacts/checkpoints/wafer_cnn_demo.pt python scripts/api_smoke_test.py
```

Main endpoints:

```text
GET  /api/v1/health
GET  /api/v1/model
POST /api/v1/predict
POST /api/v1/predict/array
GET  /api/v1/predictions
GET  /api/v1/predictions/{id}
GET  /api/v1/stats/summary
```

Read the full backend guide here:

```text
README_WEEK2_BACKEND.md
```

## v0.5 Spotfire-style simulator API

New endpoints:

```txt
POST /api/v1/simulator/run
POST /api/v1/simulator/jobs
GET  /api/v1/simulator/jobs/{job_id}
POST /api/v1/simulator/sessions
GET  /api/v1/simulator/sessions
GET  /api/v1/simulator/sessions/{session_id}
GET  /api/v1/simulator/sessions/{session_id}/export/wafers.csv
GET  /api/v1/simulator/sessions/{session_id}/export/root-causes.csv
```

The simulator creates synthetic wafer lots with controllable pattern mix, lot/tool/chamber metadata, noise level, density scale, mixed-pattern rate, yield/risk metrics, and full wafer matrices. If an image classifier or DMI feature bundle is loaded, generated wafers are routed through the real model service; otherwise the API uses a deterministic heuristic fallback so the portfolio demo never breaks.

Targeted test:

```txt
tests/test_simulator.py
```


## Enterprise v0.7 simulator hardening

The API now includes normalized simulator persistence, optional API-key protection, in-memory rate limiting, job-style simulator execution for large runs, CSV exports, configurable risk weights, and configurable severity thresholds. See `../docs/ENTERPRISE_HARDENING_V07.md`.

Validation: `PYTHONPATH=src pytest -q` -> `22 passed`.


## Enterprise v0.8 simulator performance refactor

- Added `WaferModelService.predict_batch()` for loaded-classifier inference.
- `/api/v1/simulator/run` generates the lot first, then performs one batched model prediction phase before metric rollup.
- `SimulatorRequest.model_batch_size` controls simulator inference mini-batch size.
- `SimulatorRequest.persist` now defaults to `false`; saved sessions are explicit.
- Existing normalized/capped persistence from v0.7 remains in place only for explicit saves or backward-compatible `persist=true` requests. By default, preview runs return results without writing session rows.


### v0.8 persistence defaults

`SimulatorRequest.persist` defaults to `false`. The recommended flow is:

1. `POST /api/v1/simulator/run` for preview/what-if tuning.
2. `POST /api/v1/simulator/sessions` with the selected response to save a compact analysis snapshot.

Environment knobs:

```bash
WAFERVISION_SIMULATOR_MAX_PERSIST_WAFERS=240
WAFERVISION_SIMULATOR_PERSIST_MATRIX_SIZE=40
WAFERVISION_SIMULATOR_PERSIST_DOWNSAMPLE_METHOD=area
WAFERVISION_SIMULATOR_PERSIST_MATRICES=false
WAFERVISION_SIMULATOR_MODEL_BATCH_SIZE=128
```

When matrix persistence is disabled, saved sessions store deterministic matrix seeds and regenerate compact UI matrices on load.


## v0.8 simulator performance notes

- Simulator model inference uses `WaferModelService.predict_batch()` for lot-level classifier calls.
- `/api/v1/simulator/run` defaults to preview mode and does not persist sessions unless `persist=true`.
- Explicit saves use `POST /api/v1/simulator/sessions`; SQLite stores summary + capped seeded wafer rows by default.


## Public simulator run logging

The backend creates a `simulation_runs` table for public deployment analytics. It records simulator preview runs, explicit save actions, and queued background job requests.

Stored fields:

```text
created_at, ip_hash, user_agent, scenario, wafer_count, mode, session_id
```

Raw IP addresses are not stored. The API hashes the resolved client IP with `WAFERVISION_SIMULATION_LOG_IP_SALT` using SHA-256. Set this salt to a long random value in production.
