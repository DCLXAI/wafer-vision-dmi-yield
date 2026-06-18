from __future__ import annotations

import importlib
import io
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from wafer_vision.data import ALL_CLASSES_9
from wafer_vision.model import WaferCNN


def _create_demo_checkpoint(path: Path) -> None:
    torch.manual_seed(7)
    model = WaferCNN(num_classes=len(ALL_CLASSES_9), dropout=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ALL_CLASSES_9,
            "input_size": 64,
            "model_version": "pytest-demo-checkpoint",
            "config": {"dropout": 0.25},
            "val_metrics": {"macro_f1": 0.0},
        },
        path,
    )


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    db_path = tmp_path / "history.db"
    _create_demo_checkpoint(checkpoint_path)

    monkeypatch.setenv("WAFERVISION_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("WAFERVISION_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("WAFERVISION_DEVICE", "cpu")
    monkeypatch.setenv("WAFERVISION_TOP_K", "3")

    from wafer_vision_api.settings import get_settings
    from wafer_vision_api.database import reset_database_state_for_tests

    get_settings.cache_clear()
    reset_database_state_for_tests()

    import wafer_vision_api.app as app_module

    importlib.reload(app_module)
    app = app_module.create_app()
    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()
    reset_database_state_for_tests()


def test_health_and_model_loaded(client: TestClient):
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    model = client.get("/api/v1/model")
    assert model.status_code == 200
    assert model.json()["input_size"] == 64
    assert len(model.json()["class_names"]) == 9


def test_predict_inline_array_and_history(client: TestClient):
    wafer = np.zeros((10, 10), dtype=int)
    wafer[2:8, 2:8] = 1
    wafer[4:6, 4:6] = 2

    response = client.post(
        "/api/v1/predict/array",
        json={"wafer_map": wafer.tolist(), "filename": "inline-test", "note": "pytest"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["id"] == 1
    assert "label" in body
    assert len(body["top_k"]) == 3
    assert body["input"]["rows"] == 10
    assert body["input"]["cols"] == 10

    history = client.get("/api/v1/predictions")
    assert history.status_code == 200
    assert history.json()["total"] == 1

    detail = client.get("/api/v1/predictions/1")
    assert detail.status_code == 200
    assert detail.json()["client_note"] == "pytest"

    summary = client.get("/api/v1/stats/summary")
    assert summary.status_code == 200
    assert summary.json()["total_predictions"] == 1


def test_predict_csv_upload(client: TestClient):
    wafer = np.ones((12, 12), dtype=int)
    wafer[:, 6] = 2
    payload = io.BytesIO()
    np.savetxt(payload, wafer, delimiter=",", fmt="%d")
    payload.seek(0)

    response = client.post(
        "/api/v1/predict",
        files={"file": ("wafer.csv", payload, "text/csv")},
    )
    assert response.status_code == 201
    assert response.json()["input"]["filename"] == "wafer.csv"


def test_bad_upload_extension_returns_400(client: TestClient):
    response = client.post(
        "/api/v1/predict",
        files={"file": ("wafer.txt", io.BytesIO(b"1,2,3"), "text/plain")},
    )
    assert response.status_code == 400


def test_extract_features_inline_array(client: TestClient):
    wafer = np.zeros((12, 12), dtype=int)
    wafer[2:10, 2:10] = 1
    wafer[5:7, 5:7] = 2
    response = client.post(
        "/api/v1/features/array",
        json={"wafer_map": wafer.tolist(), "filename": "features-inline"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["feature_dim"] == 59
    assert len(body["vector"]) == 59
    assert {group["name"] for group in body["groups"]} == {"region_density", "radon_mean", "radon_std", "geometry"}
