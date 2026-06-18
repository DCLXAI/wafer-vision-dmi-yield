FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    WAFERVISION_DATABASE_URL="postgresql+psycopg://wafervision:wafervision@postgres:5432/wafervision" \
    WAFERVISION_REDIS_URL="redis://redis:6379/0" \
    WAFERVISION_JOB_BACKEND="rq" \
    WAFERVISION_DEVICE=cpu

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY . .
RUN mkdir -p data/runtime artifacts/checkpoints \
    && adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "wafer_vision_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
