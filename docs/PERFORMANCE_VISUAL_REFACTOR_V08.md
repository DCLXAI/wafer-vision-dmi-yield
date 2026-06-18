# WaferVision v0.8 — Performance + Visual Refactor

## 1. Batched simulator inference

The previous simulator loop called `model_service.predict()` once per wafer. v0.8 adds `WaferModelService.predict_batch()` and changes `/api/v1/simulator/run` to generate wafer maps first, then predict them as a batch.

Files:

- `backend/src/wafer_vision_api/services/model_service.py`
- `backend/src/wafer_vision_api/routes/simulator.py`
- `backend/src/wafer_vision_api/settings.py`

Request field:

```json
{ "model_batch_size": 128 }
```

## 2. Explicit persistence

`SimulatorRequest.persist` now defaults to `false`. The cockpit exposes two actions:

- `Run preview`: no SQLite/localStorage session write.
- `Save current session`: POSTs the already-run `SimulatorResponse` to `/api/v1/simulator/sessions` and persists the capped/normalized session without rerunning the lot.

This prevents slider experimentation from writing many matrix-heavy sessions.

Files:

- `backend/src/wafer_vision_api/schemas.py`
- `frontend/src/SpotfireSimulator.tsx`
- `frontend/src/mock.ts`

## 3. Responsive browser mock mode

Browser mock simulation is now async and chunked. It yields after mode-specific chunks, or whenever a 12 ms frame budget is exceeded, using `requestAnimationFrame` when available and `setTimeout(0)` as fallback.

File:

- `frontend/src/mock.ts`

## 4. Recharts lazy loading

The main dashboard chart code moved out of `App.tsx` into a lazy component. The simulator cockpit was already lazy-loaded; now the base dashboard charts are too.

Files:

- `frontend/src/App.tsx`
- `frontend/src/components/DashboardCharts.tsx`
- `frontend/vite.config.ts`

Build chunk example:

```txt
DashboardCharts: ~3 kB
SpotfireSimulator: ~27 kB
vendor-recharts: ~378 kB
index: ~41 kB
```

## 5. Visual scanability

Severity is no longer color-only. Tiles now combine:

- severity color
- icon shape
- left-border thickness
- optional outline for critical wafers

The Canvas `animated` scanline duplicate was removed because the large wafer CSS overlay already provides the visible scanline.

Files:

- `frontend/src/components/SimulatorWidgets.tsx`
- `frontend/src/SpotfireSimulator.tsx`
- `frontend/src/styles.css`

## 6. Mobile polish

A 480px breakpoint reduces padding, tile size, chart height, top-k grid width, and large-wafer canvas size for narrow phone screens.

File:

- `frontend/src/styles.css`


## 7. Persistence storage mode

Saved sessions default to `summary_plus_seeded_rows`: capped wafer rows plus deterministic matrix seeds. This avoids writing full matrix JSON for every wafer. Compact matrices are regenerated when a saved session is loaded.

Set `WAFERVISION_SIMULATOR_PERSIST_MATRICES=true` only for demos that need stored matrix snapshots.


## 8. Verification

```bash
cd backend && PYTHONPATH=src pytest -q
# 22 passed

cd frontend && npm test -- --run
# 1 test file passed, 2 tests passed

cd frontend && npm run build
# passed; Recharts isolated into vendor-recharts and chart modules lazy-load
```
