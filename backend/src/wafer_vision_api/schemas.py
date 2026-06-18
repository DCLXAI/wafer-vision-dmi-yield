from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TopKPrediction(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class InputMetadata(BaseModel):
    filename: str | None = None
    content_type: str | None = None
    input_kind: str
    rows: int | None = None
    cols: int | None = None
    min_value: float | None = None
    max_value: float | None = None


class ModelMetadata(BaseModel):
    loaded: bool
    model_version: str
    checkpoint_path: str
    input_size: int | None = None
    class_names: list[str] = []
    device: str | None = None
    validation_macro_f1: float | None = None
    model_kind: str = "cnn"
    feature_dim: int | None = None
    feature_schema: list[str] = []
    load_error: str | None = None


class PredictResponse(BaseModel):
    id: int
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: list[TopKPrediction]
    inference_ms: float
    created_at: datetime
    input: InputMetadata
    model: ModelMetadata


class ArrayPredictRequest(BaseModel):
    wafer_map: list[list[float]]
    filename: str | None = "inline-array"
    note: str | None = None


class PredictionHistoryItem(BaseModel):
    id: int
    created_at: datetime
    filename: str | None
    input_kind: str
    predicted_label: str
    confidence: float
    inference_ms: float
    rows: int | None
    cols: int | None
    model_version: str

    model_config = {"from_attributes": True}


class PredictionDetail(BaseModel):
    id: int
    created_at: datetime
    filename: str | None
    content_type: str | None
    input_kind: str
    rows: int | None
    cols: int | None
    min_value: float | None
    max_value: float | None
    predicted_label: str
    confidence: float
    top_k: list[TopKPrediction]
    model_version: str
    checkpoint_path: str
    inference_ms: float
    client_note: str | None = None


class HistoryPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[PredictionHistoryItem]


class LabelCount(BaseModel):
    label: str
    count: int


class StatsSummary(BaseModel):
    total_predictions: int
    average_confidence: float | None
    label_counts: list[LabelCount]
    latest: list[PredictionHistoryItem]


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    model_loaded: bool
    database: str
    redis: str | None = None
    job_backend: str | None = None


class ErrorResponse(BaseModel):
    detail: str | dict[str, Any]


class FeatureGroup(BaseModel):
    name: str
    values: list[float]
    labels: list[str]


class FeatureResponse(BaseModel):
    feature_dim: int
    feature_schema: list[str]
    groups: list[FeatureGroup]
    vector: list[float]
    named_vector: dict[str, float]
    input: InputMetadata


class SimulatorRequest(BaseModel):
    wafer_count: int = Field(default=36, ge=1, le=1000)
    size: int = Field(default=64, ge=24, le=160)
    seed: int | None = Field(default=20260617)
    lot_count: int = Field(default=3, ge=1, le=32)
    noise_level: float = Field(default=0.045, ge=0.0, le=0.45)
    defect_density_scale: float = Field(default=1.0, ge=0.2, le=3.0)
    mixed_pattern_rate: float = Field(default=0.08, ge=0.0, le=0.6)
    pattern_mix: dict[str, float] | None = None
    tools: list[str] | None = None
    chambers: list[str] | None = None
    process_steps: list[str] | None = None
    persist: bool = False
    scenario_name: str | None = "balanced-baseline"
    use_model: bool = True
    performance_mode: str = Field(default="balanced", pattern="^(turbo|balanced|fidelity)$")
    return_matrix_size: int | None = Field(default=56, ge=16, le=160)
    downsample_method: str = Field(default="nearest", pattern="^(nearest|area)$")
    model_batch_size: int | None = Field(default=128, ge=1, le=512)
    risk_weights: dict[str, float] | None = None
    severity_thresholds: dict[str, float] | None = None


class SimulatorWafer(BaseModel):
    id: str
    lot_id: str
    wafer_index: int
    tool_id: str
    chamber_id: str
    process_step: str
    true_label: str
    predicted_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: list[TopKPrediction]
    defect_density: float = Field(ge=0.0, le=1.0)
    yield_estimate: float = Field(ge=0.0, le=1.0)
    edge_concentration: float = Field(ge=0.0, le=1.0)
    center_concentration: float = Field(ge=0.0, le=1.0)
    scratch_score: float = Field(ge=0.0)
    radial_non_uniformity: float = Field(ge=0.0)
    risk_score: float = Field(ge=0.0, le=100.0)
    edge_center_delta: float = 0.0
    cluster_intensity: float = 0.0
    severity: str = "Normal"
    root_cause_hint: str | None = None
    recommended_action: str | None = None
    secondary_label: str | None = None
    matrix_seed: int | None = None
    matrix: list[list[int]]


class SimulatorLabelCount(BaseModel):
    label: str
    count: int


class SimulatorToolMetric(BaseModel):
    tool_id: str
    wafers: int
    avg_risk: float
    avg_yield: float
    top_pattern: str


class SimulatorConfusionCell(BaseModel):
    true_label: str
    predicted_label: str
    count: int

class SimulatorChamberMetric(BaseModel):
    tool_id: str
    chamber_id: str
    entity_id: str
    wafers: int
    avg_risk: float
    avg_yield: float
    dominant_label: str
    excursion_score: float


class SimulatorRootCause(BaseModel):
    rank: int
    entity_type: str
    entity_id: str
    score: float
    evidence: str
    recommendation: str


class SimulatorTrendPoint(BaseModel):
    index: int
    lot_id: str
    wafer_id: str
    yield_estimate: float
    defect_density: float
    risk_score: float
    predicted_label: str


class SimulatorSummary(BaseModel):
    total_wafers: int
    avg_yield: float
    avg_confidence: float
    avg_defect_density: float
    high_risk_count: int
    p95_risk: float = 0.0
    yield_loss_pp: float = 0.0
    model_agreement: float | None = None
    simulation_runtime_ms: float | None = None
    label_counts: list[SimulatorLabelCount]
    true_label_counts: list[SimulatorLabelCount]
    tool_risk: list[SimulatorToolMetric]
    chamber_risk: list[SimulatorChamberMetric] = Field(default_factory=list)
    root_causes: list[SimulatorRootCause] = Field(default_factory=list)
    confusion: list[SimulatorConfusionCell]
    trend: list[SimulatorTrendPoint]


class SimulatorResponse(BaseModel):
    session_id: str
    created_at: datetime
    params: SimulatorRequest
    wafers: list[SimulatorWafer]
    summary: SimulatorSummary
    model: ModelMetadata
    persisted_wafer_count: int | None = None
    persistence_note: str | None = None


class SimulatorSessionListItem(BaseModel):
    session_id: str
    created_at: datetime
    scenario_name: str | None
    wafer_count: int
    persisted_wafer_count: int | None = None
    avg_yield: float | None = None
    high_risk_count: int | None = None


class SimulatorSessionPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[SimulatorSessionListItem]


class SimulatorJobStatus(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    progress: float = Field(ge=0.0, le=1.0)
    session_id: str | None = None
    error: str | None = None
    result: SimulatorResponse | None = None
    backend: str | None = None
    queue_name: str | None = None
    external_job_id: str | None = None
    result_available: bool | None = None
