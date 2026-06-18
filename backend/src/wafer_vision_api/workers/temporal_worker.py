from __future__ import annotations

import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from wafer_vision_api.database import init_db
from wafer_vision_api.jobs.temporal_workflows import SimulatorWorkflow, run_simulator_activity
from wafer_vision_api.settings import get_settings


async def amain() -> None:
    settings = get_settings()
    init_db(settings.database_url)
    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[SimulatorWorkflow],
        activities=[run_simulator_activity],
    )
    await worker.run()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
