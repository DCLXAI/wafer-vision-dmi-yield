from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow


@activity.defn
async def run_simulator_activity(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    from wafer_vision_api.workers.simulator_tasks import execute_simulator_job

    return await asyncio.to_thread(execute_simulator_job, job_id, payload)


@workflow.defn
class SimulatorWorkflow:
    @workflow.run
    async def run(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await workflow.execute_activity(
            run_simulator_activity,
            job_id,
            payload,
            start_to_close_timeout=timedelta(minutes=20),
        )
