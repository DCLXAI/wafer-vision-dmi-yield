from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from wafer_vision_api.app import create_app
from wafer_vision_api.database import reset_database_state_for_tests
from wafer_vision_api.settings import Settings, get_settings


def test_inline_job_backend_persists_durable_job(tmp_path: Path, monkeypatch):
    db_url = f"sqlite:///{tmp_path}/jobs.db"
    monkeypatch.setenv("WAFERVISION_DATABASE_URL", db_url)
    monkeypatch.setenv("WAFERVISION_JOB_BACKEND", "inline")
    monkeypatch.setenv("WAFERVISION_RATE_LIMIT_BACKEND", "memory")
    get_settings.cache_clear()
    reset_database_state_for_tests()
    settings = Settings(
        database_url=db_url,
        job_backend="inline",
        rate_limit_backend="memory",
        checkpoint_path=tmp_path / "missing.pt",
        cors_origins="http://localhost:5173",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        start = client.post("/api/v1/simulator/jobs", json={"wafer_count": 6, "size": 32, "seed": 9, "use_model": False})
        assert start.status_code == 200, start.text
        job_id = start.json()["job_id"]
        status = None
        for _ in range(40):
            res = client.get(f"/api/v1/simulator/jobs/{job_id}")
            assert res.status_code == 200, res.text
            status = res.json()
            if status["status"] == "succeeded":
                break
            time.sleep(0.1)
        assert status is not None
        assert status["status"] == "succeeded"
        assert status["session_id"]
        assert status["result"] is None
        assert status["backend"] == "inline"
        assert status["queue_name"] == "local-thread"
        session = client.get(f"/api/v1/simulator/sessions/{status['session_id']}")
        assert session.status_code == 200, session.text
        assert session.json()["summary"]["total_wafers"] == 6
