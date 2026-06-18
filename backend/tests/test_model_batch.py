from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wafer_vision.data import ALL_CLASSES_9
from wafer_vision.model import WaferCNN
from wafer_vision_api.services.model_service import WaferModelService


def _create_checkpoint(path: Path) -> None:
    torch.manual_seed(13)
    model = WaferCNN(num_classes=len(ALL_CLASSES_9), dropout=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ALL_CLASSES_9,
            "input_size": 64,
            "model_version": "batch-test-checkpoint",
            "config": {"dropout": 0.25},
        },
        path,
    )


def test_cnn_predict_batch_returns_one_output_per_wafer(tmp_path):
    checkpoint = tmp_path / "cnn.pt"
    _create_checkpoint(checkpoint)
    service = WaferModelService(checkpoint, device="cpu", top_k=3, model_kind="cnn")
    service.load()
    wafers = []
    for offset in range(5):
        wafer = np.ones((32, 32), dtype=np.int64)
        wafer[8 + offset : 16 + offset, 8:16] = 2
        wafers.append(wafer)
    outputs = service.predict_batch(wafers, batch_size=4)
    assert len(outputs) == len(wafers)
    assert all(len(item.top_k) == 3 for item in outputs)
    assert all(0 <= item.confidence <= 1 for item in outputs)
