from __future__ import annotations

from fastapi.testclient import TestClient

from wafer_vision.simulator import calculate_spatial_metrics, generate_wafer_map
from wafer_vision_api.app import create_app
from wafer_vision_api.database import reset_database_state_for_tests
from wafer_vision_api.settings import Settings


def test_generate_wafer_map_and_metrics():
    wafer = generate_wafer_map("Edge-Ring", size=48, noise_level=0.02)
    assert wafer.shape == (48, 48)
    assert set(wafer.ravel().tolist()).issubset({0, 1, 2})
    metrics = calculate_spatial_metrics(wafer, confidence=0.9)
    assert 0 <= metrics.defect_density <= 1
    assert 0 <= metrics.yield_estimate <= 1
    assert 0 <= metrics.risk_score <= 100


def test_simulator_run_works_without_loaded_model(tmp_path, monkeypatch):
    monkeypatch.setenv("WAFERVISION_DATABASE_URL", f"sqlite:///{tmp_path}/sim.db")
    reset_database_state_for_tests()
    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/sim.db",
        checkpoint_path=tmp_path / "missing.pt",
        cors_origins="http://localhost:5173",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        res = client.post("/api/v1/simulator/run", json={"wafer_count": 8, "size": 32, "seed": 7, "persist": True})
        assert res.status_code == 200, res.text
        payload = res.json()
        assert payload["summary"]["total_wafers"] == 8
        assert len(payload["wafers"]) == 8
        assert payload["wafers"][0]["matrix"]
        sessions = client.get("/api/v1/simulator/sessions")
        assert sessions.status_code == 200
        assert sessions.json()["total"] >= 1
        loaded = client.get(f"/api/v1/simulator/sessions/{payload['session_id']}")
        assert loaded.status_code == 200
        assert loaded.json()["session_id"] == payload["session_id"]
