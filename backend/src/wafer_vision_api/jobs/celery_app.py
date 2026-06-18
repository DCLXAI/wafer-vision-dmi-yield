from __future__ import annotations

from celery import Celery

from wafer_vision_api.settings import get_settings
from wafer_vision_api.workers.simulator_tasks import execute_simulator_job

settings = get_settings()

celery_app = Celery(
    "wafer_vision_api",
    broker=settings.effective_celery_broker_url,
    backend=settings.effective_celery_result_backend,
)
celery_app.conf.update(
    task_default_queue=settings.celery_queue_name,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    result_expires=settings.job_result_ttl_seconds,
    task_time_limit=settings.job_timeout_seconds,
)


@celery_app.task(name="wafer_vision_api.simulator.run", bind=True)
def run_simulator_celery_task(self, job_id: str, payload: dict):
    return execute_simulator_job(job_id, payload)
