from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from wafer_vision_api.settings import get_settings

_DATABASE_URL_OVERRIDE: str | None = None


class Base(DeclarativeBase):
    pass


def is_sqlite_url(database_url: str) -> bool:
    return database_url.startswith("sqlite")


def ensure_sqlite_parent_dir(database_url: str) -> None:
    if not is_sqlite_url(database_url):
        return
    path_part = database_url.replace("sqlite:///", "", 1)
    if path_part == ":memory:":
        return
    Path(path_part).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    database_url = _DATABASE_URL_OVERRIDE or settings.database_url
    ensure_sqlite_parent_dir(database_url)
    if is_sqlite_url(database_url):
        return create_engine(database_url, connect_args={"check_same_thread": False}, future=True)
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=int(settings.db_pool_size),
        max_overflow=int(settings.db_max_overflow),
        pool_timeout=int(settings.db_pool_timeout_seconds),
        pool_recycle=int(settings.db_pool_recycle_seconds),
        future=True,
    )


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autocommit=False, autoflush=False, future=True, expire_on_commit=False)


def init_db(database_url: str | None = None) -> None:
    global _DATABASE_URL_OVERRIDE
    if database_url is not None and database_url != _DATABASE_URL_OVERRIDE:
        _DATABASE_URL_OVERRIDE = database_url
        get_engine.cache_clear()
        get_sessionmaker.cache_clear()
    from wafer_vision_api import db_models  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _run_sqlite_lightweight_migrations(engine)


def _run_sqlite_lightweight_migrations(engine: Engine) -> None:
    if not str(engine.url).startswith("sqlite"):
        return
    with engine.begin() as conn:
        tables = {row[0] for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        if "simulation_sessions" in tables:
            columns = {row[1] for row in conn.execute(text("PRAGMA table_info(simulation_sessions)"))}
            migrations = {
                "persisted_wafer_count": "ALTER TABLE simulation_sessions ADD COLUMN persisted_wafer_count INTEGER DEFAULT 0",
                "matrix_persist_size": "ALTER TABLE simulation_sessions ADD COLUMN matrix_persist_size INTEGER",
                "response_payload_kind": "ALTER TABLE simulation_sessions ADD COLUMN response_payload_kind VARCHAR(40) DEFAULT 'summary_plus_rows'",
                "model_json": "ALTER TABLE simulation_sessions ADD COLUMN model_json TEXT",
            }
            for column, ddl in migrations.items():
                if column not in columns:
                    conn.execute(text(ddl))
        if "simulator_jobs" in tables:
            columns = {row[1] for row in conn.execute(text("PRAGMA table_info(simulator_jobs)"))}
            migrations = {
                "external_job_id": "ALTER TABLE simulator_jobs ADD COLUMN external_job_id VARCHAR(160)",
                "queue_name": "ALTER TABLE simulator_jobs ADD COLUMN queue_name VARCHAR(120)",
                "result_json": "ALTER TABLE simulator_jobs ADD COLUMN result_json TEXT",
            }
            for column, ddl in migrations.items():
                if column not in columns:
                    conn.execute(text(ddl))


def get_db() -> Generator[Session, None, None]:
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()


def reset_database_state_for_tests() -> None:
    global _DATABASE_URL_OVERRIDE
    _DATABASE_URL_OVERRIDE = None
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
