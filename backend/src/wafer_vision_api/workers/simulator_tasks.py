from __future__ import annotations

import json
import traceback
from typing import Any

from wafer_vision_api.database import get_sessionmaker, init_db
from wafer_vision_api.jobs.store import update_job_record
from wafer_vision_api.schemas import SimulatorRequest
from wafer_vision_api.services.model_service import WaferModelService
from wafer_vision_api.settings import get_settings


def execute_simulator_job(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run one simulator job inside a worker process.

    RQ, Celery, Temporal, and the inline backend all call this same function.
    The worker always persists a compact normalized session so polling can return
    a small job status and the frontend can fetch `/simulator/sessions/{id}`.
    """
    settings = get_settings()
    init_db(settings.database_url)
    update_job_record(job_id, status="running", progress=0.10, started=True)
    try:
        request = SimulatorRequest.model_validate(payload)
        # Background jobs should create a durable analysis session even when the
        # original UI request was a preview. This is still compact because
        # `_persist_session` stores capped rows and seeds by default.
        request = request.model_copy(update={"persist": True})
        model_service = WaferModelService(
            checkpoint_path=settings.checkpoint_path,
            device=settings.device,
            top_k=settings.top_k,
            model_kind=settings.model_kind,
        )
        model_service.load()

        from wafer_vision_api.routes.simulator import _persist_session, _run

        response = _run(request, model_service, settings)
        session_factory = get_sessionmaker()
        with session_factory() as db:
            persisted, note = _persist_session(db, response, settings)
        response = response.model_copy(update={"persisted_wafer_count": persisted, "persistence_note": note})
        result_payload = response.model_dump(mode="json")
        update_job_record(
            job_id,
            status="succeeded",
            progress=1.0,
            session_id=response.session_id,
            result_payload=result_payload if settings.job_poll_persist_results else None,
            finished=True,
        )
        return result_payload
    except Exception as exc:  # pragma: no cover - worker failure path depends on runtime
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        update_job_record(job_id, status="failed", progress=1.0, error=error, finished=True)
        raise


def run_simulator_job(job_id: str, payload_json: str | dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
    return execute_simulator_job(job_id, payload)
