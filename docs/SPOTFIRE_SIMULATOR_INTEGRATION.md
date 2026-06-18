# Spotfire-Style Wafermap Pattern Recognition Simulator Integration

This version turns WaferVision from a single-wafer classifier into an interactive wafer-map analytics cockpit inspired by Spotfire-style visual analysis workflows.

## What was added

### Backend

New FastAPI routes:

```txt
POST /api/v1/simulator/run
GET  /api/v1/simulator/sessions
GET  /api/v1/simulator/sessions/{session_id}
```

The simulator creates synthetic wafer lots with:

- controllable pattern mix
- wafer count, map size, random seed
- noise level
- defect-density scale
- mixed-pattern rate
- lot/tool/chamber/process metadata
- model prediction through the loaded classifier when available
- heuristic fallback when the model is not loaded
- SQLite-backed simulation session persistence

New backend modules:

```txt
backend/src/wafer_vision/simulator.py
backend/src/wafer_vision_api/routes/simulator.py
backend/tests/test_simulator.py
```

### Frontend

New React component:

```txt
frontend/src/SpotfireSimulator.tsx
```

The UI adds:

- scenario controls
- saved simulation sessions
- KPI summary
- pattern distribution chart
- tool/chamber risk Pareto
- yield/risk trend chart
- defect-density × confidence scatter plot
- wafer-map trellis grid
- dynamic filters by predicted label, tool, and risk band
- selected wafer detail panel
- top-k model confidence bars
- process-cause explanation panel

## Scenario presets

```txt
balanced-baseline
edge-ring-excursion
scratch-handling-event
noisy-mixed-lot
```

Each scenario controls the defect pattern mix and default noise/density/mixed-pattern parameters.

## Run backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src uvicorn wafer_vision_api.app:app --reload --host 0.0.0.0 --port 8000
```

The simulator works even when no model artifact is loaded, because it falls back to a deterministic heuristic predictor. With a real `.pt` image checkpoint or `.joblib` DMI feature bundle, it routes generated wafer maps through the model service.

## Run frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Use demo mode for a fully browser-only presentation:

```txt
VITE_USE_MOCKS=true
```

Use real API mode for FastAPI integration:

```txt
VITE_USE_MOCKS=false
VITE_API_BASE_URL=http://localhost:8000
```

## Why this matters for portfolio review

The project now demonstrates more than image classification. It shows a complete visual analytics product loop:

```txt
Synthetic wafer lot generation
→ spatial pattern recognition
→ model confidence analysis
→ tool/process risk exploration
→ wafer trellis marking
→ selected wafer root-cause explanation
→ saved analysis sessions
```

This is much closer to the kind of workflow a process/yield engineer would expect from an industrial analytics dashboard.
