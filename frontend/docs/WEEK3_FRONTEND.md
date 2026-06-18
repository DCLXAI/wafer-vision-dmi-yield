# Week 3 Frontend Implementation Notes

## Goal

Build a portfolio-grade React frontend for WaferVision:

1. Upload wafer map file
2. Preview die-level heatmap in the browser
3. Send file or array to Week 2 FastAPI
4. Show AI prediction, confidence, top-k labels, and latency
5. Refresh analytics and prediction history

## Why this frontend is stronger than a static demo

- It has a real API boundary and can call Week 2 FastAPI directly.
- It has a mock API so the demo still works on Vercel without a Python backend.
- It parses wafer maps client-side for instant heatmap feedback before inference finishes.
- It renders three useful analytics views: class volume, class share, and recent confidence trend.
- It explains likely defect causes and next operator checks, which shows domain understanding.

## API contract

The frontend expects the Week 2 backend contract:

```txt
GET  /api/v1/health
GET  /api/v1/model
POST /api/v1/predict
POST /api/v1/predict/array
GET  /api/v1/stats/summary
GET  /api/v1/predictions?limit=50&offset=0&label=Scratch
```

## Environment modes

Frontend-only demo:

```bash
VITE_USE_MOCKS=true
```

Real backend:

```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCKS=false
```

## UX flow

```txt
File/sample chosen
  ↓
Browser parses preview matrix
  ↓
Heatmap renders immediately
  ↓
User clicks Run AI classification
  ↓
FastAPI or mock API returns PredictResponse
  ↓
Prediction card + Recharts + history table refresh
```

## Files that matter most

- `src/App.tsx` — dashboard composition and state flow
- `src/api.ts` — real FastAPI client
- `src/mock.ts` — local demo client
- `src/parsers.ts` — CSV / NPY / image preview
- `src/wafer.ts` — domain labels and synthetic samples
- `src/styles.css` — dark cinematic inspection console styling

## Next step for Week 4

- Add README screenshots or a short screen recording.
- Deploy frontend to Vercel.
- Deploy backend to Render or Railway.
- Put real model checkpoint path in backend `.env`.
- Add portfolio case-study page with model metrics from Week 1.
