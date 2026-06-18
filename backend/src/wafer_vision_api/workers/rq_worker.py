from __future__ import annotations

from redis import Redis
from rq import Queue, Worker

from wafer_vision_api.database import init_db
from wafer_vision_api.settings import get_settings


def main() -> None:
    settings = get_settings()
    init_db(settings.database_url)
    redis_conn = Redis.from_url(settings.redis_url)
    queues = [Queue(settings.job_queue_name, connection=redis_conn)]
    worker = Worker(queues, connection=redis_conn)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
