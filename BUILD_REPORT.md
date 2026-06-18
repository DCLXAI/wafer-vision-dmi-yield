# WaferVision Enterprise Simulator v0.9 — Build Report

## Scope

v0.9 replaces the remaining demo-grade persistence and background-job pieces with a production-style worker plane:

- Postgres is now the default durable database for predictions, simulator sessions, wafer rows, and simulator job status.
- Redis/RQ is the default background simulator queue.
- Celery and Temporal adapters are available behind the same `/api/v1/simulator/jobs` contract.
- Redis-backed rate limiting can be shared across API replicas, with memory fallback for tests.
- SQLite and inline jobs remain only as explicit laptop/test fallbacks.

v0.8 performance work is preserved: batched simulator model inference, preview/save separation, compact seeded session persistence, chunked mock generation, Canvas trellis rendering, and lazy chart chunks.

## Verification

Backend:

```bash
cd backend
PYTHONPATH=src pytest -q
# 24 passed
```

Frontend:

```bash
cd frontend
npm test -- --run
# 1 test file passed, 2 tests passed

npm run build
# passed
```

Latest frontend production chunk check:

```txt
DashboardCharts:      3.09 kB
SimulatorCharts:      4.17 kB
SpotfireSimulator:   25.10 kB
App index:           41.38 kB
vendor-react:       192.67 kB
vendor-recharts:    378.20 kB
```

## Runtime matrix

| Mode | Database | Queue | Rate limiter | Use case |
|---|---|---|---|---|
| production default | Postgres | Redis/RQ | Redis | normal deployed demo/product |
| Celery adapter | Postgres | Redis/Celery | Redis | teams already using Celery |
| Temporal adapter | Postgres | Temporal server | Redis | long-running workflow environments |
| offline fallback | SQLite | inline thread | memory | tests or laptop-only demos |

## Production notes

- `WAFERVISION_JOB_POLL_PERSIST_RESULTS=false` keeps job polling payloads small. Workers persist compact sessions and return `session_id`; the frontend fetches `/api/v1/simulator/sessions/{session_id}`.
- `Base.metadata.create_all()` is intentionally kept for portfolio/local bootstrapping. Use Alembic migrations before real multi-tenant deployment.
- Simulator risk weights remain empirical and should be calibrated with real fab/yield data before operational use.
- Temporal support assumes a running Temporal server; the Compose profile includes a starter `temporalio/auto-setup` service for local experimentation.

See `docs/POSTGRES_REDIS_WORKERS_V09.md` for commands and environment variables.
