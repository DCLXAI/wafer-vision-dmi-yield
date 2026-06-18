from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from wafer_vision_api.app import create_app
from wafer_vision_api.database import get_sessionmaker, reset_database_state_for_tests
from wafer_vision_api.db_models import SimulationRunLog
from wafer_vision_api.settings import Settings


def test_simulator_run_records_privacy_preserving_log(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path}/run_logs.db"
    monkeypatch.setenv("WAFERVISION_DATABASE_URL", database_url)
    reset_database_state_for_tests()
    settings = Settings(
        database_url=database_url,
        checkpoint_path=tmp_path / "missing.pt",
        cors_origins="http://localhost:5173",
        rate_limit_enabled=False,
        simulation_log_ip_salt="test-salt",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        res = client.post(
            "/api/v1/simulator/run",
            json={"wafer_count": 6, "size": 32, "seed": 31, "scenario_name": "edge-ring-excursion"},
            headers={"x-forwarded-for": "203.0.113.17, 10.0.0.1", "user-agent": "pytest-agent/1.0"},
        )
        assert res.status_code == 200, res.text
        session_id = res.json()["session_id"]

    with get_sessionmaker()() as db:
        row = db.query(SimulationRunLog).one()

    assert row.created_at is not None
    assert row.ip_hash == hashlib.sha256("test-salt:203.0.113.17".encode("utf-8")).hexdigest()
    assert "203.0.113.17" not in row.ip_hash
    assert row.user_agent == "pytest-agent/1.0"
    assert row.scenario == "edge-ring-excursion"
    assert row.wafer_count == 6
    assert row.mode == "preview"
    assert row.session_id == session_id
