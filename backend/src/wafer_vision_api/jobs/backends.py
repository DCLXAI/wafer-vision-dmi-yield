from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Any

from wafer_vision_api.jobs.store import create_job_record, get_job_record, make_job_id, status_from_record, update_job_record
from wafer_vision_api.schemas import SimulatorJobStatus, SimulatorRequest
from wafer_vision_api.settings import Settings
from wafer_vision_api.redis_client import get_rq_redis_connection


class JobBackendError(RuntimeError):
    pass


class SimulatorJobBackend(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    async def enqueue(self, payload: SimulatorRequest) -> SimulatorJobStatus:
        raise NotImplementedError

    @abstractmethod
    async def get(self, job_id: str) -> SimulatorJobStatus | None:
        raise NotImplementedError


class InlineJobBackend(SimulatorJobBackend):
    """Local dev/test backend.

    It still writes durable Postgres/SQLite job rows, but execution runs in a
    daemon thread in the current process. Do not use this for multi-worker prod.
    """

    async def enqueue(self, payload: SimulatorRequest) -> SimulatorJobStatus:
        return await asyncio.to_thread(self._enqueue_sync, payload)

    def _enqueue_sync(self, payload: SimulatorRequest) -> SimulatorJobStatus:
        from wafer_vision_api.workers.simulator_tasks import execute_simulator_job

        job_id = make_job_id()
        payload_dict = payload.model_dump(mode="json")
        record = create_job_record(
            job_id=job_id,
            backend="inline",
            queue_name="local-thread",
            request_payload=payload_dict,
            external_job_id=job_id,
        )

        def runner() -> None:
            execute_simulator_job(job_id, payload_dict)

        threading.Thread(target=runner, daemon=True, name=f"wafervision-{job_id}").start()
        return status_from_record(record)  # type: ignore[return-value]

    async def get(self, job_id: str) -> SimulatorJobStatus | None:
        return await asyncio.to_thread(lambda: status_from_record(get_job_record(job_id)))


class RQJobBackend(SimulatorJobBackend):
    async def enqueue(self, payload: SimulatorRequest) -> SimulatorJobStatus:
        return await asyncio.to_thread(self._enqueue_sync, payload)

    def _enqueue_sync(self, payload: SimulatorRequest) -> SimulatorJobStatus:
        try:
            from rq import Queue
        except ImportError as exc:  # pragma: no cover
            raise JobBackendError("Install redis and rq to use WAFERVISION_JOB_BACKEND=rq.") from exc

        job_id = make_job_id()
        payload_dict = payload.model_dump(mode="json")
        # Important: create the Postgres job row before publishing to Redis.
        # Fast workers can otherwise start before the polling ledger exists.
        record = create_job_record(
            job_id=job_id,
            backend="rq",
            queue_name=self.settings.job_queue_name,
            request_payload=payload_dict,
            external_job_id=job_id,
        )
        try:
            redis_conn = get_rq_redis_connection(self.settings)
            queue = Queue(self.settings.job_queue_name, connection=redis_conn)
            job = queue.enqueue(
                "wafer_vision_api.workers.simulator_tasks.run_simulator_job",
                job_id,
                payload_dict,
                job_id=job_id,
                job_timeout=int(self.settings.job_timeout_seconds),
                result_ttl=int(self.settings.job_result_ttl_seconds),
                failure_ttl=int(self.settings.job_failure_ttl_seconds),
                description=f"WaferVision simulator {payload.wafer_count} wafers / {payload.performance_mode}",
            )
            if job.id != record.external_job_id:
                record = update_job_record(job_id, external_job_id=job.id) or record
        except Exception as exc:
            update_job_record(job_id, status="failed", progress=1.0, error=f"Could not enqueue RQ job: {exc}", finished=True)
            raise
        return status_from_record(record)  # type: ignore[return-value]

    async def get(self, job_id: str) -> SimulatorJobStatus | None:
        return await asyncio.to_thread(self._get_sync, job_id)

    def _get_sync(self, job_id: str) -> SimulatorJobStatus | None:
        record = get_job_record(job_id)
        if record is None:
            return None
        result_payload: dict[str, Any] | None = None
        try:
            from rq.job import Job

            redis_conn = get_rq_redis_connection(self.settings)
            rq_job = Job.fetch(record.external_job_id or job_id, connection=redis_conn)
            rq_status = str(rq_job.get_status(refresh=True))
            mapped = {
                "queued": "queued",
                "deferred": "queued",
                "scheduled": "queued",
                "started": "running",
                "finished": "succeeded",
                "failed": "failed",
                "stopped": "failed",
                "canceled": "cancelled",
            }.get(rq_status, record.status)
            if mapped != record.status:
                record = update_job_record(job_id, status=mapped, progress=1.0 if mapped in {"succeeded", "failed", "cancelled"} else record.progress) or record
            if rq_status == "finished":
                try:
                    value = rq_job.return_value(refresh=True)
                except TypeError:  # older RQ fallback
                    value = rq_job.result
                if isinstance(value, dict):
                    result_payload = value
                    if self.settings.job_poll_persist_results and not record.result_json:
                        record = update_job_record(job_id, result_payload=value, session_id=value.get("session_id"), status="succeeded", progress=1.0, finished=True) or record
                    elif value.get("session_id") and not record.session_id:
                        record = update_job_record(job_id, session_id=value.get("session_id"), status="succeeded", progress=1.0, finished=True) or record
            elif rq_status == "failed" and not record.error:
                record = update_job_record(job_id, status="failed", progress=1.0, error=rq_job.exc_info or "RQ job failed.", finished=True) or record
        except Exception:
            pass
        return status_from_record(record, result_payload=result_payload if self.settings.job_poll_persist_results else None)


class CeleryJobBackend(SimulatorJobBackend):
    async def enqueue(self, payload: SimulatorRequest) -> SimulatorJobStatus:
        return await asyncio.to_thread(self._enqueue_sync, payload)

    def _enqueue_sync(self, payload: SimulatorRequest) -> SimulatorJobStatus:
        try:
            from wafer_vision_api.jobs.celery_app import celery_app
        except ImportError as exc:  # pragma: no cover
            raise JobBackendError("Install celery to use WAFERVISION_JOB_BACKEND=celery.") from exc

        job_id = make_job_id()
        payload_dict = payload.model_dump(mode="json")
        record = create_job_record(
            job_id=job_id,
            backend="celery",
            queue_name=self.settings.celery_queue_name,
            request_payload=payload_dict,
            external_job_id=job_id,
        )
        try:
            async_result = celery_app.send_task(
                "wafer_vision_api.simulator.run",
                args=[job_id, payload_dict],
                task_id=job_id,
                queue=self.settings.celery_queue_name,
            )
            if async_result.id != record.external_job_id:
                record = update_job_record(job_id, external_job_id=async_result.id) or record
        except Exception as exc:
            update_job_record(job_id, status="failed", progress=1.0, error=f"Could not enqueue Celery job: {exc}", finished=True)
            raise
        return status_from_record(record)  # type: ignore[return-value]

    async def get(self, job_id: str) -> SimulatorJobStatus | None:
        return await asyncio.to_thread(self._get_sync, job_id)

    def _get_sync(self, job_id: str) -> SimulatorJobStatus | None:
        record = get_job_record(job_id)
        if record is None:
            return None
        try:
            from celery.result import AsyncResult
            from wafer_vision_api.jobs.celery_app import celery_app

            result = AsyncResult(record.external_job_id or job_id, app=celery_app)
            state = str(result.state).upper()
            mapped = {"PENDING": "queued", "STARTED": "running", "SUCCESS": "succeeded", "FAILURE": "failed", "RETRY": "running", "REVOKED": "cancelled"}.get(state, record.status)
            result_payload = result.result if state == "SUCCESS" and isinstance(result.result, dict) else None
            if mapped != record.status or result_payload:
                record = update_job_record(
                    job_id,
                    status=mapped,
                    progress=1.0 if mapped in {"succeeded", "failed", "cancelled"} else record.progress,
                    result_payload=result_payload if (self.settings.job_poll_persist_results and result_payload) else None,
                    session_id=result_payload.get("session_id") if result_payload else None,
                    error=str(result.result) if state == "FAILURE" else None,
                    finished=mapped in {"succeeded", "failed", "cancelled"},
                ) or record
            return status_from_record(record, result_payload=result_payload if self.settings.job_poll_persist_results else None)
        except Exception:
            return status_from_record(record)


class TemporalJobBackend(SimulatorJobBackend):
    async def enqueue(self, payload: SimulatorRequest) -> SimulatorJobStatus:
        try:
            from temporalio.client import Client
            from wafer_vision_api.jobs.temporal_workflows import SimulatorWorkflow
        except ImportError as exc:  # pragma: no cover
            raise JobBackendError("Install temporalio to use WAFERVISION_JOB_BACKEND=temporal.") from exc

        job_id = make_job_id()
        payload_dict = payload.model_dump(mode="json")
        record = create_job_record(
            job_id=job_id,
            backend="temporal",
            queue_name=self.settings.temporal_task_queue,
            request_payload=payload_dict,
            external_job_id=job_id,
        )
        client = await Client.connect(self.settings.temporal_address, namespace=self.settings.temporal_namespace)
        await client.start_workflow(
            SimulatorWorkflow.run,
            job_id,
            payload_dict,
            id=job_id,
            task_queue=self.settings.temporal_task_queue,
        )
        return status_from_record(record)  # type: ignore[return-value]

    async def get(self, job_id: str) -> SimulatorJobStatus | None:
        return await asyncio.to_thread(lambda: status_from_record(get_job_record(job_id)))


def get_job_backend(settings: Settings) -> SimulatorJobBackend:
    backend = settings.normalized_job_backend
    if backend in {"inline", "thread", "local"}:
        return InlineJobBackend(settings)
    if backend == "rq":
        return RQJobBackend(settings)
    if backend == "celery":
        return CeleryJobBackend(settings)
    if backend == "temporal":
        return TemporalJobBackend(settings)
    raise JobBackendError(f"Unknown WAFERVISION_JOB_BACKEND={settings.job_backend!r}.")
