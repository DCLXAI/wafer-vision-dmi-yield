from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from wafer_vision_api.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PredictionRecord(Base):
    """Stores one model inference result for dashboard analytics."""

    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    input_kind: Mapped[str] = mapped_column(String(50), default="upload")
    rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cols: Mapped[int | None] = mapped_column(Integer, nullable=True)
    min_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    predicted_label: Mapped[str] = mapped_column(String(80), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    top_k_json: Mapped[str] = mapped_column(Text)

    model_version: Mapped[str] = mapped_column(String(120), default="unknown")
    checkpoint_path: Mapped[str] = mapped_column(Text)
    inference_ms: Mapped[float] = mapped_column(Float)
    client_note: Mapped[str | None] = mapped_column(Text, nullable=True)


class SimulationSession(Base):
    """Stores one synthetic wafer-map simulator run.

    v0.9 keeps normalized session metadata and capped wafer rows in Postgres.
    Matrix snapshots remain optional; deterministic seeds regenerate compact
    matrices when the cockpit opens a saved session.
    """

    __tablename__ = "simulation_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    scenario_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    wafer_count: Mapped[int] = mapped_column(Integer)
    persisted_wafer_count: Mapped[int] = mapped_column(Integer, default=0)
    matrix_persist_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_payload_kind: Mapped[str] = mapped_column(String(40), default="summary_plus_seeded_rows")
    params_json: Mapped[str] = mapped_column(Text)
    summary_json: Mapped[str] = mapped_column(Text)
    model_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Kept nullable for backward compatibility with older SQLite files.
    response_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    wafers: Mapped[list["SimulationWaferRecord"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="SimulationWaferRecord.wafer_index",
    )


class SimulationWaferRecord(Base):
    """Stores one persisted wafer from a simulator session."""

    __tablename__ = "simulation_wafers"
    __table_args__ = (UniqueConstraint("session_id", "wafer_id", name="uq_simulation_wafers_session_wafer"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(80), ForeignKey("simulation_sessions.session_id", ondelete="CASCADE"), index=True)
    wafer_id: Mapped[str] = mapped_column(String(120), index=True)
    wafer_index: Mapped[int] = mapped_column(Integer, index=True)
    lot_id: Mapped[str] = mapped_column(String(80), index=True)
    tool_id: Mapped[str] = mapped_column(String(80), index=True)
    chamber_id: Mapped[str] = mapped_column(String(20), index=True)
    process_step: Mapped[str] = mapped_column(String(80), index=True)
    true_label: Mapped[str] = mapped_column(String(80), index=True)
    predicted_label: Mapped[str] = mapped_column(String(80), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    risk_score: Mapped[float] = mapped_column(Float, index=True)
    severity: Mapped[str] = mapped_column(String(40), index=True)
    wafer_json: Mapped[str] = mapped_column(Text)
    matrix_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    session: Mapped[SimulationSession] = relationship(back_populates="wafers")


class SimulationRunLog(Base):
    """Privacy-preserving ledger of simulator executions.

    The request IP is never stored in raw form. `ip_hash` is a salted SHA-256
    digest derived at request time so public deployments can track repeated
    simulator usage without collecting direct IP addresses.
    """

    __tablename__ = "simulation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    ip_hash: Mapped[str] = mapped_column(String(64), index=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    scenario: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    wafer_count: Mapped[int] = mapped_column(Integer, index=True)
    mode: Mapped[str] = mapped_column(String(40), index=True)
    session_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)


class SimulatorJobRecord(Base):
    """Durable simulator job state shared by API and worker processes.

    Redis/RQ/Celery/Temporal own execution. Postgres owns the polling ledger so
    job state survives API restarts and multiple API replicas can read it.
    """

    __tablename__ = "simulator_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    external_job_id: Mapped[str | None] = mapped_column(String(160), nullable=True, index=True)
    backend: Mapped[str] = mapped_column(String(40), default="rq", index=True)
    queue_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    request_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
