# WaferVision v0.9 — Postgres + Redis/RQ/Celery/Temporal worker plane

## What changed

The simulator no longer depends on SQLite plus in-memory job state for production-style runs.

```txt
FastAPI API
  ↓ creates durable simulator_jobs row
Postgres
  ↓ publishes job id + request payload
Redis/RQ by default, Celery optional, Temporal optional
  ↓ worker executes simulator lot generation + batch model inference
Postgres
  ↓ normalized simulation_sessions + simulation_wafers rows
React cockpit
  ↓ polls job status, then fetches saved session by session_id
```

## Defaults

- `WAFERVISION_DATABASE_URL=postgresql+psycopg://...`
- `WAFERVISION_REDIS_URL=redis://localhost:6379/0`
- `WAFERVISION_JOB_BACKEND=rq`
- `WAFERVISION_JOB_POLL_PERSIST_RESULTS=false`

`false` is important: job polling stays small. The worker persists a compact session and returns `session_id`; the frontend then calls `/api/v1/simulator/sessions/{session_id}`.

## Start the production-style local stack

```bash
docker compose up --build postgres redis api rq-worker
```

Then open:

```txt
http://localhost:8000/api/v1/health
http://localhost:8000/docs
```

## Celery adapter

```bash
WAFERVISION_JOB_BACKEND=celery \
  docker compose --profile celery up --build postgres redis api celery-worker
```

## Temporal adapter

```bash
WAFERVISION_JOB_BACKEND=temporal WAFERVISION_TEMPORAL_ADDRESS=temporal:7233 \
  docker compose --profile temporal up --build postgres redis api temporal temporal-worker
```

## Offline fallback

For tests or a one-process laptop demo:

```bash
WAFERVISION_DATABASE_URL=sqlite:///./data/runtime/wafervision.db
WAFERVISION_JOB_BACKEND=inline
WAFERVISION_RATE_LIMIT_BACKEND=memory
```

## Design notes

- Postgres owns durable truth: predictions, sessions, wafer rows, and simulator job ledger.
- Redis owns ephemeral coordination: RQ queue and distributed rate-limit buckets.
- RQ is the simplest default worker backend. Celery is included for teams that already run Celery. Temporal is included for long-running, retry-heavy workflow environments.
- Matrix snapshots are not stored by default. Saved sessions keep deterministic matrix seeds and regenerate compact UI matrices.
