from __future__ import annotations

from wafer_vision_api.routes.simulator import _run
from wafer_vision_api.schemas import ModelMetadata, SimulatorRequest, TopKPrediction
from wafer_vision_api.settings import Settings


class FakeBatchModelService:
    def __init__(self) -> None:
        self.batch_calls: list[tuple[int, int]] = []

    def predict_batch(self, wafer_maps, batch_size: int = 128):
        self.batch_calls.append((len(wafer_maps), batch_size))
        return [
            type(
                "PredictionOutput",
                (),
                {
                    "label": "Edge-Ring",
                    "confidence": 0.91,
                    "top_k": [TopKPrediction(label="Edge-Ring", probability=0.91)],
                },
            )()
            for _ in wafer_maps
        ]

    def metadata(self):
        return ModelMetadata(
            loaded=True,
            model_version="fake-batched-model",
            checkpoint_path="memory://fake",
            class_names=["Edge-Ring"],
            model_kind="cnn",
        )


def test_simulator_uses_one_batched_model_call_for_lot() -> None:
    service = FakeBatchModelService()
    request = SimulatorRequest(wafer_count=17, size=32, seed=2026, use_model=True, performance_mode="balanced", model_batch_size=11)

    response = _run(request, service, Settings(checkpoint_path="missing.pt"))  # type: ignore[arg-type]

    assert response.summary.total_wafers == 17
    assert service.batch_calls == [(17, 11)]
    assert {wafer.predicted_label for wafer in response.wafers} == {"Edge-Ring"}
