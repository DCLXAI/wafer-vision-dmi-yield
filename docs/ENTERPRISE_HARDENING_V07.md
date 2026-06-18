# WaferVision Enterprise Hardening v0.7

This release upgrades the v0.6 Spotfire-style simulator into a more production-shaped industrial analytics cockpit. The goal is not only better visuals, but safer persistence, bounded runtime behavior, cleaner React structure, and a clearer explanation path for process-engineering metrics.

## What changed

| # | Review item | v0.7 implementation |
|---:|---|---|
| 1 | Huge `response_json` blob | Added normalized `simulation_wafers` table. Sessions now persist metadata, summary/model JSON, and a capped wafer subset. Large matrices are downsampled before persistence. Legacy `response_json` remains nullable for backward compatibility. |
| 2 | Blocking simulator endpoint | `POST /api/v1/simulator/run` now offloads CPU work via FastAPI threadpool. Added job-style endpoints for large runs: `POST /api/v1/simulator/jobs` and `GET /api/v1/simulator/jobs/{job_id}`. |
| 3 | Magic risk constants | Introduced named `DEFAULT_RISK_WEIGHTS`, configurable `risk_weights`, and comments that label them as empirical simulator weights, not calibrated fab rules. |
| 4 | Python cluster loop | `_cluster_intensity` now uses `scipy.ndimage.uniform_filter` when SciPy is available, with a NumPy fallback. |
| 5 | No auth | Added optional `x-api-key` protection through middleware. Health/model metadata remain public; stateful and heavy endpoints can be protected by `WAFERVISION_API_KEY`. |
| 6 | Large React components | Split common simulator UI into `components/SimulatorWidgets.tsx`, formatters into `utils/format.ts`, export helpers into `utils/export.ts`, and session data loading into `hooks/useSimulatorSessions.ts`. |
| 7 | Confusion matrix unused | Added an interactive confusion matrix panel to the simulator cockpit. |
| 8 | No rate limiting | Added in-memory rate limiting middleware with a stricter simulator bucket. This is suitable for portfolio/demo deployment; production should swap it for Redis or gateway-level limits. |
| 9 | Recharts chunk warning | Lazy-loaded the simulator route area and split Recharts with Vite `manualChunks`. |
| 10 | Duplicate utility functions | Moved `pct`, `pp`, `risk`, `ms`, `dt`, and `compact` into `src/utils/format.ts`. |
| 11 | Session pagination unused | Added session pagination controls using backend `limit/offset/total`. |
| 12 | Hardcoded severity bands | Added configurable `severity_thresholds` in simulator request and centralized threshold normalization. |
| 13 | Health router double registration unclear | Added comments in `app.py` explaining public root health and versioned API health registration. |
| 14 | Nearest-only downsample | Added `area` downsampling option for fidelity/persisted matrices. |
| 15 | Single-stage Dockerfile | Replaced backend Dockerfile with a multi-stage build and non-root runtime user. |
| 16 | Static mock fixture drift | Added `npm run snapshot:mock`, which can regenerate a simulator snapshot from the real backend contract. |
| 17 | No React Error Boundary | Added `ErrorBoundary` around the simulator cockpit. |
| 18 | No export | Added CSV exports for wafer rows and root-cause queue in the backend, plus frontend CSV/JSON export actions. |

## New API surface

```txt
POST /api/v1/simulator/jobs
GET  /api/v1/simulator/jobs/{job_id}
GET  /api/v1/simulator/sessions/{session_id}/export/wafers.csv
GET  /api/v1/simulator/sessions/{session_id}/export/root-causes.csv
```

The existing API remains compatible:

```txt
POST /api/v1/simulator/run
GET  /api/v1/simulator/sessions
GET  /api/v1/simulator/sessions/{session_id}
```

## New configuration

```bash
WAFERVISION_API_KEY=
WAFERVISION_RATE_LIMIT_ENABLED=true
WAFERVISION_RATE_LIMIT_WINDOW_SECONDS=60
WAFERVISION_RATE_LIMIT_REQUESTS=240
WAFERVISION_SIMULATOR_RATE_LIMIT_REQUESTS=45
WAFERVISION_SIMULATOR_MAX_PERSIST_WAFERS=240
WAFERVISION_SIMULATOR_PERSIST_MATRIX_SIZE=40
WAFERVISION_SIMULATOR_PERSIST_DOWNSAMPLE_METHOD=area
```

## Validation

```txt
Backend:  PYTHONPATH=src pytest -q  -> 19 passed
Frontend: npm run build             -> passed
Frontend: npm test -- --run         -> 1 file passed, 2 tests passed
```

## Remaining production notes

The in-memory job store and in-memory rate limiter are intentionally lightweight. For real multi-worker production, replace them with Redis + RQ/Celery, Temporal, or a managed queue. The simulator risk weights are now configurable and documented, but still empirical; a real fab deployment would calibrate them against historical yield loss, metrology, and tool maintenance records.
