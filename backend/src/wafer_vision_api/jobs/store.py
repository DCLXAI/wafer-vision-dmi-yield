from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from wafer_vision_api.database import get_sessionmaker
from wafer_vision_api.db_models import SimulatorJobRecord
from wafer_vision_api.schemas import SimulatorJobStatus, SimulatorResponse

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_job_id() -> str:
    import uuid

    return f"job-{utc_now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"


def create_job_record(
    *,
    job_id: str,
    backend: str,
    queue_name: str | None,
    request_payload: dict[str, Any],
    external_job_id: str | None = None,
    db: Session | None = None,
) -> SimulatorJobRecord:
    own_session = db is None
    session = db or get_sessionmaker()()
    try:
        now = utc_now()
        record = SimulatorJobRecord(
            job_id=job_id,
            external_job_id=external_job_id,
            backend=backend,
            queue_name=queue_name,
            status="queued",
            progress=0.0,
            created_at=now,
            updated_at=now,
            request_json=json.dumps(request_payload, ensure_ascii=False),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def update_job_record(
    job_id: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    session_id: str | None = None,
    result_payload: dict[str, Any] | None = None,
    error: str | None = None,
    external_job_id: str | None = None,
    started: bool = False,
    finished: bool = False,
    db: Session | None = None,
) -> SimulatorJobRecord | None:
    own_session = db is None
    session = db or get_sessionmaker()()
    try:
        record = session.query(SimulatorJobRecord).filter(SimulatorJobRecord.job_id == job_id).one_or_none()
        if record is None:
            return None
        now = utc_now()
        if status is not None:
            record.status = status
        if progress is not None:
            record.progress = max(0.0, min(1.0, float(progress)))
        if session_id is not None:
            record.session_id = session_id
        if result_payload is not None:
            record.result_json = json.dumps(result_payload, ensure_ascii=False)
        if error is not None:
            record.error = error
        if external_job_id is not None:
            record.external_job_id = external_job_id
        if started and record.started_at is None:
            record.started_at = now
        if finished:
            record.finished_at = now
        record.updated_at = now
        session.commit()
        session.refresh(record)
        return record
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def get_job_record(job_id: str, db: Session | None = None) -> SimulatorJobRecord | None:
    own_session = db is None
    session = db or get_sessionmaker()()
    try:
        return session.query(SimulatorJobRecord).filter(SimulatorJobRecord.job_id == job_id).one_or_none()
    finally:
        if own_session:
            session.close()


def status_from_record(record: SimulatorJobRecord | None, *, result_payload: dict[str, Any] | None = None) -> SimulatorJobStatus | None:
    if record is None:
        return None
    payload = result_payload
    if payload is None and record.result_json:
        try:
            payload = json.loads(record.result_json)
        except Exception:
            payload = None
    result = None
    if isinstance(payload, dict):
        try:
            result = SimulatorResponse.model_validate(payload)
        except Exception:
            result = None
    return SimulatorJobStatus(
        job_id=record.job_id,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        progress=float(record.progress or 0.0),
        session_id=record.session_id,
        error=record.error,
        result=result,
        backend=record.backend,
        queue_name=record.queue_name,
        external_job_id=record.external_job_id,
        result_available=bool(result is not None or record.result_json),
    )
