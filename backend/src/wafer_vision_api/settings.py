from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - compatibility fallback
    from pydantic import BaseSettings  # type: ignore

    SettingsConfigDict = dict  # type: ignore


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env.

    v0.9 production default is PostgreSQL + Redis/RQ. SQLite and the inline job
    backend remain supported for tests and laptop-only demos.
    """

    app_name: str = "WaferVision API"
    api_prefix: str = "/api/v1"
    environment: str = "local"

    checkpoint_path: Path = Path("artifacts/checkpoints/wafer_cnn_best.pt")
    model_kind: str = "auto"  # auto | cnn | kaggle_svm

    database_url: str = "postgresql+psycopg://wafervision:wafervision@localhost:5432/wafervision"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout_seconds: int = 30
    db_pool_recycle_seconds: int = 1800

    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "wafervision"
    redis_socket_timeout_seconds: float = 2.0
    redis_health_check_interval_seconds: int = 30

    job_backend: str = "rq"  # inline | rq | celery | temporal
    job_queue_name: str = "wafervision-simulator"
    job_timeout_seconds: int = 900
    job_result_ttl_seconds: int = 3600
    job_failure_ttl_seconds: int = 86400
    job_poll_persist_results: bool = False

    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    celery_queue_name: str = "wafervision-simulator"

    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "wafervision-simulator"

    device: str = "auto"  # auto | cpu | cuda | mps
    top_k: int = 5
    max_upload_mb: int = 10
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"

    # Optional demo/enterprise guardrails. Leave API key empty for local demo mode.
    api_key: str | None = None
    rate_limit_enabled: bool = True
    rate_limit_backend: str = "redis"  # redis | memory
    rate_limit_fail_open: bool = True
    rate_limit_window_seconds: int = 60
    rate_limit_requests: int = 240
    simulator_rate_limit_requests: int = 45

    # Simulator persistence guardrails: never store 1000 full matrices in one JSON blob.
    simulator_max_persist_wafers: int = 240
    simulator_persist_matrix_size: int = 40
    simulator_persist_downsample_method: str = "area"
    simulator_persist_matrices: bool = False
    simulator_model_batch_size: int = 128

    # Public analytics: record simulator executions without storing raw IPs.
    simulation_run_logging_enabled: bool = True
    simulation_log_ip_salt: str = "wafervision-simulation-runs"

    model_config = SettingsConfigDict(  # type: ignore[assignment]
        env_file=".env",
        env_prefix="WAFERVISION_",
        extra="ignore",
    )

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb) * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def normalized_job_backend(self) -> str:
        value = self.job_backend.lower().strip()
        return "inline" if value in {"local", "thread"} else value

    @property
    def normalized_rate_limit_backend(self) -> str:
        value = self.rate_limit_backend.lower().strip()
        return value if value in {"redis", "memory"} else "memory"

    @property
    def effective_celery_broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def effective_celery_result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
