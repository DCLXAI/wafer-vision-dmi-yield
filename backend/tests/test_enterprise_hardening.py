from __future__ import annotations

from fastapi.testclient import TestClient

from wafer_vision.simulator import downsample_matrix, generate_wafer_map, severity_band
from wafer_vision_api.app import create_app
from wafer_vision_api.database import reset_database_state_for_tests
from wafer_vision_api.settings import Settings


def test_simulator_persistence_is_capped_and_rehydrated(tmp_path, monkeypatch):
    monkeypatch.setenv("WAFERVISION_DATABASE_URL", f"sqlite:///{tmp_path}/sim.db")
    reset_database_state_for_tests()
    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/sim.db",
        checkpoint_path=tmp_path / "missing.pt",
        simulator_max_persist_wafers=5,
        simulator_persist_matrix_size=24,
        cors_origins="http://localhost:5173",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        res = client.post("/api/v1/simulator/run", json={"wafer_count": 12, "size": 48, "seed": 11, "persist": True})
        assert res.status_code == 200, res.text
        payload = res.json()
        assert payload["summary"]["total_wafers"] == 12
        assert payload["persisted_wafer_count"] == 5
        assert "Persisted first 5" in payload["persistence_note"]

        loaded = client.get(f"/api/v1/simulator/sessions/{payload['session_id']}")
        assert loaded.status_code == 200, loaded.text
        body = loaded.json()
        assert body["summary"]["total_wafers"] == 12
        assert len(body["wafers"]) == 5
        assert len(body["wafers"][0]["matrix"]) == 24

        wafers_csv = client.get(f"/api/v1/simulator/sessions/{payload['session_id']}/export/wafers.csv")
        assert wafers_csv.status_code == 200, wafers_csv.text
        assert "wafer_id,lot_id" in wafers_csv.text
        assert "root_cause_hint" in wafers_csv.text

        root_csv = client.get(f"/api/v1/simulator/sessions/{payload['session_id']}/export/root-causes.csv")
        assert root_csv.status_code == 200, root_csv.text
        assert "rank,entity_type" in root_csv.text


def test_optional_api_key_protects_stateful_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("WAFERVISION_DATABASE_URL", f"sqlite:///{tmp_path}/auth.db")
    reset_database_state_for_tests()
    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/auth.db",
        checkpoint_path=tmp_path / "missing.pt",
        api_key="secret-key",
        cors_origins="http://localhost:5173",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
        blocked = client.get("/api/v1/simulator/sessions")
        assert blocked.status_code == 401
        allowed = client.get("/api/v1/simulator/sessions", headers={"x-api-key": "secret-key"})
        assert allowed.status_code == 200


def test_configurable_risk_helpers_and_area_downsample():
    wafer = generate_wafer_map("Scratch", size=48, noise_level=0.02)
    area = downsample_matrix(wafer, 16, method="area")
    nearest = downsample_matrix(wafer, 16, method="nearest")
    assert area.shape == (16, 16)
    assert nearest.shape == (16, 16)
    assert severity_band(50, {"monitor": 10, "warning": 40, "critical": 80}) == "Warning"


def test_simulator_preview_run_does_not_persist_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("WAFERVISION_DATABASE_URL", f"sqlite:///{tmp_path}/preview.db")
    reset_database_state_for_tests()
    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/preview.db",
        checkpoint_path=tmp_path / "missing.pt",
        cors_origins="http://localhost:5173",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        before = client.get("/api/v1/simulator/sessions")
        assert before.status_code == 200
        before_total = before.json()["total"]
        res = client.post("/api/v1/simulator/run", json={"wafer_count": 10, "size": 40, "seed": 17})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["persisted_wafer_count"] == 0
        assert "disabled" in body["persistence_note"].lower()
        sessions = client.get("/api/v1/simulator/sessions")
        assert sessions.status_code == 200
        assert sessions.json()["total"] == before_total
