from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session

from wafer_vision.simulator import (
    LABELS,
    calculate_spatial_metrics,
    choose_metadata,
    downsample_matrix,
    generate_wafer_map,
    normalize_pattern_mix,
    normalize_severity_thresholds,
    root_cause_hint,
    severity_band,
)
from wafer_vision_api.database import get_db
from wafer_vision_api.db_models import SimulationRunLog, SimulationSession, SimulationWaferRecord
from wafer_vision_api.dependencies import get_app_settings, get_model_service
from wafer_vision_api.schemas import (
    ModelMetadata,
    SimulatorChamberMetric,
    SimulatorConfusionCell,
    SimulatorJobStatus,
    SimulatorLabelCount,
    SimulatorRequest,
    SimulatorResponse,
    SimulatorRootCause,
    SimulatorSessionListItem,
    SimulatorSessionPage,
    SimulatorSummary,
    SimulatorToolMetric,
    SimulatorTrendPoint,
    SimulatorWafer,
    TopKPrediction,
)
from wafer_vision_api.jobs.backends import JobBackendError, get_job_backend
from wafer_vision_api.services.model_service import ModelNotLoadedError, WaferModelService
from wafer_vision_api.settings import Settings

router = APIRouter(prefix="/simulator", tags=["enterprise wafermap simulator"])



@router.post("/run", response_model=SimulatorResponse)
async def run_simulator(
    request: Request,
    payload: SimulatorRequest,
    db: Session = Depends(get_db),
    model_service: WaferModelService = Depends(get_model_service),
    settings: Settings = Depends(get_app_settings),
) -> SimulatorResponse:
    """Run a simulator request without blocking the event loop.

    v0.8 defaults to persist=false so tuning sliders and repeated what-if runs do
    not write every transient result to SQLite. Set persist=true for backward
    compatibility or use POST /simulator/sessions for an explicit Save action.
    """
    response = await run_in_threadpool(_run, payload, model_service, settings)
    if payload.persist:
        persisted, note = await run_in_threadpool(_persist_session, db, response, settings)
        final_response = response.model_copy(update={"persisted_wafer_count": persisted, "persistence_note": note})
        _log_simulation_run(db, request, settings, payload, mode="persist", session_id=final_response.session_id)
        return final_response
    final_response = response.model_copy(update={"persisted_wafer_count": 0, "persistence_note": "Persistence disabled for preview run. Use Save analysis to persist a compact session."})
    _log_simulation_run(db, request, settings, payload, mode="preview", session_id=final_response.session_id)
    return final_response


@router.post("/jobs", response_model=SimulatorJobStatus)
async def start_simulator_job(
    request: Request,
    payload: SimulatorRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> SimulatorJobStatus:
    """Create a durable background simulator job.

    v0.9 replaces the in-memory job dictionary with a provider-backed queue.
    Default production path is Redis + RQ with durable Postgres job rows; Celery
    and Temporal adapters can be selected without changing the frontend API.
    """
    try:
        status = await get_job_backend(settings).enqueue(payload)
        _log_simulation_run(db, request, settings, payload, mode="job_queued", session_id=status.session_id)
        return status
    except JobBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not enqueue simulator job: {exc}") from exc


@router.get("/jobs/{job_id}", response_model=SimulatorJobStatus)
async def get_simulator_job(
    job_id: str,
    settings: Settings = Depends(get_app_settings),
) -> SimulatorJobStatus:
    try:
        status = await get_job_backend(settings).get(job_id)
    except JobBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if status is None:
        raise HTTPException(status_code=404, detail="Simulator job not found.")
    return status


@router.get("/sessions", response_model=SimulatorSessionPage)
def list_sessions(limit: int = 20, offset: int = 0, db: Session = Depends(get_db)) -> SimulatorSessionPage:
    limit = max(1, min(100, int(limit)))
    offset = max(0, int(offset))
    total = db.query(SimulationSession).count()
    rows = db.query(SimulationSession).order_by(desc(SimulationSession.created_at)).offset(offset).limit(limit).all()
    items: list[SimulatorSessionListItem] = []
    for row in rows:
        try:
            summary = json.loads(row.summary_json)
        except Exception:
            summary = {}
        items.append(
            SimulatorSessionListItem(
                session_id=row.session_id,
                created_at=row.created_at,
                scenario_name=row.scenario_name,
                wafer_count=row.wafer_count,
                persisted_wafer_count=getattr(row, "persisted_wafer_count", None),
                avg_yield=_safe_float(summary.get("avg_yield")),
                high_risk_count=int(summary.get("high_risk_count", 0)) if isinstance(summary, dict) else None,
            )
        )
    return SimulatorSessionPage(total=total, limit=limit, offset=offset, items=items)


@router.post("/sessions", response_model=SimulatorResponse)
async def save_session(
    request: Request,
    response: SimulatorResponse,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> SimulatorResponse:
    """Explicitly save the current analysis session.

    The frontend now separates Run from Save. This prevents accidental DB growth
    while users scrub scenario sliders, and stores only normalized/capped rows
    rather than a huge response_json blob.
    """
    persisted, note = await run_in_threadpool(_persist_session, db, response, settings)
    final_response = response.model_copy(update={"persisted_wafer_count": persisted, "persistence_note": note})
    _log_simulation_run(db, request, settings, final_response.params, mode="save", session_id=final_response.session_id)
    return final_response


@router.get("/sessions/{session_id}", response_model=SimulatorResponse)
def get_session(session_id: str, db: Session = Depends(get_db)) -> SimulatorResponse:
    row = db.query(SimulationSession).filter(SimulationSession.session_id == session_id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Simulation session not found.")
    response = _rehydrate_session(row, db)
    if response is not None:
        return response
    # Backward compatibility for v0.6 databases that only had response_json.
    try:
        return SimulatorResponse.model_validate(json.loads(row.response_json or "{}"))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Stored simulation payload is corrupted: {exc}") from exc


@router.get("/sessions/{session_id}/export/wafers.csv")
def export_session_wafers(session_id: str, db: Session = Depends(get_db)) -> StreamingResponse:
    row = db.query(SimulationSession).filter(SimulationSession.session_id == session_id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Simulation session not found.")
    wafer_rows = db.query(SimulationWaferRecord).filter(SimulationWaferRecord.session_id == session_id).order_by(SimulationWaferRecord.wafer_index).all()
    headers = [
        "wafer_id",
        "lot_id",
        "wafer_index",
        "tool_id",
        "chamber_id",
        "process_step",
        "true_label",
        "predicted_label",
        "confidence",
        "yield_estimate",
        "defect_density",
        "risk_score",
        "severity",
        "root_cause_hint",
        "recommended_action",
    ]
    rows: list[dict[str, Any]] = []
    for item in wafer_rows:
        payload = json.loads(item.wafer_json)
        rows.append({key: payload.get(key) for key in headers})
    return _csv_response(headers, rows, f"{session_id}-wafers.csv")


@router.get("/sessions/{session_id}/export/root-causes.csv")
def export_session_root_causes(session_id: str, db: Session = Depends(get_db)) -> StreamingResponse:
    row = db.query(SimulationSession).filter(SimulationSession.session_id == session_id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Simulation session not found.")
    try:
        summary = json.loads(row.summary_json)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stored summary payload is corrupted: {exc}") from exc
    headers = ["rank", "entity_type", "entity_id", "score", "evidence", "recommendation"]
    rows = [{key: item.get(key) for key in headers} for item in summary.get("root_causes", [])]
    return _csv_response(headers, rows, f"{session_id}-root-causes.csv")


def _log_simulation_run(db: Session, request: Request, settings: Settings, payload: SimulatorRequest, *, mode: str, session_id: str | None) -> None:
    if not settings.simulation_run_logging_enabled:
        return
    try:
        db.add(
            SimulationRunLog(
                ip_hash=_hash_request_ip(request, settings),
                user_agent=(request.headers.get("user-agent") or "")[:512] or None,
                scenario=payload.scenario_name,
                wafer_count=int(payload.wafer_count),
                mode=mode,
                session_id=session_id,
            )
        )
        db.commit()
    except Exception:
        db.rollback()


def _hash_request_ip(request: Request, settings: Settings) -> str:
    ip = _client_ip(request)
    salt = settings.simulation_log_ip_salt or settings.api_key or settings.app_name
    return hashlib.sha256(f"{salt}:{ip}".encode("utf-8")).hexdigest()


def _client_ip(request: Request) -> str:
    for header in ("cf-connecting-ip", "x-real-ip"):
        value = request.headers.get(header)
        if value:
            return value.strip()
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def _run(payload: SimulatorRequest, model_service: WaferModelService, settings: Settings | None = None) -> SimulatorResponse:
    started = time.perf_counter()
    rng = np.random.default_rng(payload.seed)
    labels, weights = normalize_pattern_mix(payload.pattern_mix)
    created_at = datetime.now(timezone.utc)
    session_id = f"sim-{created_at.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    lot_count = max(1, min(payload.lot_count, payload.wafer_count))
    lots = [f"LOT-{session_id[-4:].upper()}-{i + 1:02d}" for i in range(lot_count)]

    if payload.performance_mode == "turbo":
        return_matrix_size = min(payload.return_matrix_size or payload.size, 40)
        downsample_method = "nearest"
    elif payload.performance_mode == "fidelity":
        return_matrix_size = payload.return_matrix_size or payload.size
        downsample_method = payload.downsample_method or "area"
    else:
        return_matrix_size = payload.return_matrix_size or 56
        downsample_method = payload.downsample_method or "nearest"

    generated: list[dict[str, Any]] = []
    for idx in range(payload.wafer_count):
        true_label = str(rng.choice(labels, p=weights))
        secondary = None
        if payload.mixed_pattern_rate > 0 and rng.random() < payload.mixed_pattern_rate:
            candidates = [label for label in LABELS if label not in {true_label, "None"}]
            secondary = str(rng.choice(candidates)) if candidates else None

        matrix_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        matrix = generate_wafer_map(
            true_label,
            size=payload.size,
            noise_level=payload.noise_level,
            defect_density_scale=payload.defect_density_scale,
            rng=np.random.default_rng(matrix_seed),
            secondary_label=secondary,
        )
        tool, chamber, step = choose_metadata(true_label, idx, rng, payload.tools, payload.chambers, payload.process_steps)
        generated.append(
            {
                "idx": idx,
                "true_label": true_label,
                "secondary": secondary,
                "matrix_seed": matrix_seed,
                "matrix": matrix,
                "tool": tool,
                "chamber": chamber,
                "step": step,
            }
        )

    should_use_model = bool(payload.use_model) and payload.performance_mode != "turbo"
    predictions = _predict_batch_or_fallback(
        [np.asarray(item["matrix"]) for item in generated],
        [str(item["true_label"]) for item in generated],
        payload,
        model_service,
        rng,
        settings=settings,
        use_model=should_use_model,
    )

    wafers: list[SimulatorWafer] = []
    for item, prediction in zip(generated, predictions, strict=True):
        idx = int(item["idx"])
        matrix = np.asarray(item["matrix"])
        true_label = str(item["true_label"])
        tool = str(item["tool"])
        chamber = str(item["chamber"])
        step = str(item["step"])
        metrics = calculate_spatial_metrics(matrix, prediction["confidence"], risk_weights=payload.risk_weights)
        wafer_id = f"{lots[idx % lot_count]}-W{idx + 1:03d}"
        cause_title, action = root_cause_hint(str(prediction["label"]), tool, chamber)
        ui_matrix = downsample_matrix(matrix, return_matrix_size, method=downsample_method)
        wafers.append(
            SimulatorWafer(
                id=wafer_id,
                lot_id=lots[idx % lot_count],
                wafer_index=idx + 1,
                tool_id=tool,
                chamber_id=chamber,
                process_step=step,
                true_label=true_label,
                predicted_label=str(prediction["label"]),
                confidence=float(prediction["confidence"]),
                top_k=prediction["top_k"],
                defect_density=metrics.defect_density,
                yield_estimate=metrics.yield_estimate,
                edge_concentration=metrics.edge_concentration,
                center_concentration=metrics.center_concentration,
                scratch_score=metrics.scratch_score,
                radial_non_uniformity=metrics.radial_non_uniformity,
                risk_score=metrics.risk_score,
                edge_center_delta=metrics.edge_center_delta,
                cluster_intensity=metrics.cluster_intensity,
                severity=severity_band(metrics.risk_score, payload.severity_thresholds),
                root_cause_hint=cause_title,
                recommended_action=action,
                secondary_label=item["secondary"],
                matrix_seed=int(item["matrix_seed"]),
                matrix=ui_matrix.astype(int).tolist(),
            )
        )

    runtime_ms = (time.perf_counter() - started) * 1000.0
    metadata = model_service.metadata() if model_service else ModelMetadata(
        loaded=False,
        model_version="heuristic-simulator",
        checkpoint_path="none",
        class_names=LABELS,
        model_kind="simulator-fallback",
    )
    return SimulatorResponse(
        session_id=session_id,
        created_at=created_at,
        params=payload,
        wafers=wafers,
        summary=_summarize(wafers, runtime_ms=runtime_ms, severity_thresholds=payload.severity_thresholds),
        model=metadata,
    )


def _predict_batch_or_fallback(
    matrices: list[np.ndarray],
    true_labels: list[str],
    payload: SimulatorRequest,
    model_service: WaferModelService,
    rng: np.random.Generator,
    *,
    settings: Settings | None = None,
    use_model: bool,
) -> list[dict[str, Any]]:
    if use_model and matrices:
        try:
            default_batch_size = int(settings.simulator_model_batch_size) if settings is not None else 128
            batch_size = int(payload.model_batch_size or (64 if payload.performance_mode == "fidelity" else default_batch_size))
            outputs = model_service.predict_batch(matrices, batch_size=batch_size)
            return [{"label": output.label, "confidence": output.confidence, "top_k": output.top_k} for output in outputs]
        except (ModelNotLoadedError, Exception):
            pass
    return [_fallback_prediction(label, payload, rng) for label in true_labels]


def _fallback_prediction(true_label: str, payload: SimulatorRequest, rng: np.random.Generator) -> dict[str, Any]:
    confusion_map = {
        "Edge-Ring": ["Edge-Loc", "Donut", "Loc"],
        "Edge-Loc": ["Edge-Ring", "Loc", "Scratch"],
        "Center": ["Loc", "Donut", "Random"],
        "Donut": ["Center", "Edge-Ring", "Loc"],
        "Loc": ["Random", "Center", "Edge-Loc"],
        "Near-full": ["Edge-Ring", "Random", "Loc"],
        "Random": ["Loc", "None", "Scratch"],
        "Scratch": ["Loc", "Edge-Loc", "Random"],
        "None": ["Random", "Loc", "Center"],
    }
    difficulty = min(
        0.52,
        payload.noise_level * 0.75
        + payload.mixed_pattern_rate * 0.45
        + max(0, payload.defect_density_scale - 1.0) * 0.08
        + (0.06 if payload.performance_mode == "turbo" else 0.0),
    )
    predicted = true_label
    if rng.random() < difficulty:
        predicted = str(rng.choice(confusion_map.get(true_label, ["Random"])))
    base = 0.92 - difficulty * 0.55 + rng.normal(0.0, 0.025)
    if predicted != true_label:
        base -= 0.12
    first = float(np.clip(base, 0.52, 0.98))
    alternatives = [x for x in [*confusion_map.get(predicted, []), *LABELS] if x != predicted]
    labels = [predicted, *list(dict.fromkeys(alternatives))[:4]]
    raw = np.asarray([first, 0.16, 0.08, 0.04, 0.025], dtype=np.float64)[: len(labels)]
    raw = raw / raw.sum()
    top_k = [TopKPrediction(label=label, probability=float(raw[i])) for i, label in enumerate(labels)]
    return {"label": predicted, "confidence": top_k[0].probability, "top_k": top_k}


def _summarize(wafers: list[SimulatorWafer], runtime_ms: float | None = None, severity_thresholds: dict[str, float] | None = None) -> SimulatorSummary:
    total = max(1, len(wafers))
    warning_threshold = normalize_severity_thresholds(severity_thresholds)["warning"]
    pred_counts = Counter(w.predicted_label for w in wafers)
    true_counts = Counter(w.true_label for w in wafers)
    confusion = Counter((w.true_label, w.predicted_label) for w in wafers)
    tool_groups: dict[str, list[SimulatorWafer]] = defaultdict(list)
    chamber_groups: dict[tuple[str, str], list[SimulatorWafer]] = defaultdict(list)
    for wafer in wafers:
        tool_groups[wafer.tool_id].append(wafer)
        chamber_groups[(wafer.tool_id, wafer.chamber_id)].append(wafer)

    tool_risk: list[SimulatorToolMetric] = []
    for tool_id, rows in tool_groups.items():
        top_pattern = Counter(w.predicted_label for w in rows).most_common(1)[0][0]
        tool_risk.append(
            SimulatorToolMetric(
                tool_id=tool_id,
                wafers=len(rows),
                avg_risk=float(sum(w.risk_score for w in rows) / max(1, len(rows))),
                avg_yield=float(sum(w.yield_estimate for w in rows) / max(1, len(rows))),
                top_pattern=top_pattern,
            )
        )
    tool_risk.sort(key=lambda item: item.avg_risk, reverse=True)

    chamber_risk: list[SimulatorChamberMetric] = []
    root_causes: list[SimulatorRootCause] = []
    for (tool_id, chamber_id), rows in chamber_groups.items():
        dominant_label = Counter(w.predicted_label for w in rows).most_common(1)[0][0]
        avg_risk = float(sum(w.risk_score for w in rows) / max(1, len(rows)))
        avg_yield = float(sum(w.yield_estimate for w in rows) / max(1, len(rows)))
        high_frac = sum(1 for w in rows if w.risk_score >= warning_threshold) / max(1, len(rows))
        excursion_score = float(np.clip(avg_risk * 0.72 + high_frac * 32 + np.log1p(len(rows)) * 2.2, 0.0, 100.0))
        chamber_risk.append(
            SimulatorChamberMetric(
                tool_id=tool_id,
                chamber_id=chamber_id,
                entity_id=f"{tool_id}-{chamber_id}",
                wafers=len(rows),
                avg_risk=avg_risk,
                avg_yield=avg_yield,
                dominant_label=dominant_label,
                excursion_score=excursion_score,
            )
        )
        title, action = root_cause_hint(dominant_label, tool_id, chamber_id)
        root_causes.append(
            SimulatorRootCause(
                rank=0,
                entity_type="tool_chamber",
                entity_id=f"{tool_id}-{chamber_id}",
                score=excursion_score,
                evidence=f"{len(rows)} wafers, avg risk {avg_risk:.1f}, dominant pattern {dominant_label}, high-risk share {high_frac:.0%}.",
                recommendation=action or title,
            )
        )
    chamber_risk.sort(key=lambda item: item.excursion_score, reverse=True)
    root_causes.sort(key=lambda item: item.score, reverse=True)
    root_causes = [item.model_copy(update={"rank": i + 1}) for i, item in enumerate(root_causes[:6])]

    risks = np.asarray([w.risk_score for w in wafers], dtype=np.float64)
    avg_yield = float(sum(w.yield_estimate for w in wafers) / total)
    agreement = sum(1 for w in wafers if w.true_label == w.predicted_label) / total if wafers else None
    return SimulatorSummary(
        total_wafers=len(wafers),
        avg_yield=avg_yield,
        avg_confidence=float(sum(w.confidence for w in wafers) / total),
        avg_defect_density=float(sum(w.defect_density for w in wafers) / total),
        high_risk_count=sum(1 for w in wafers if w.risk_score >= warning_threshold),
        p95_risk=float(np.percentile(risks, 95)) if risks.size else 0.0,
        yield_loss_pp=float((1.0 - avg_yield) * 100.0),
        model_agreement=float(agreement) if agreement is not None else None,
        simulation_runtime_ms=float(runtime_ms) if runtime_ms is not None else None,
        label_counts=[SimulatorLabelCount(label=k, count=v) for k, v in pred_counts.most_common()],
        true_label_counts=[SimulatorLabelCount(label=k, count=v) for k, v in true_counts.most_common()],
        tool_risk=tool_risk,
        chamber_risk=chamber_risk[:24],
        root_causes=root_causes,
        confusion=[SimulatorConfusionCell(true_label=k[0], predicted_label=k[1], count=v) for k, v in confusion.most_common()],
        trend=[
            SimulatorTrendPoint(
                index=i + 1,
                lot_id=w.lot_id,
                wafer_id=w.id,
                yield_estimate=w.yield_estimate,
                defect_density=w.defect_density,
                risk_score=w.risk_score,
                predicted_label=w.predicted_label,
            )
            for i, w in enumerate(wafers)
        ],
    )


def _persist_session(db: Session, response: SimulatorResponse, settings: Settings) -> tuple[int, str]:
    # Idempotent save: replacing an existing session avoids unique-key failures
    # when the user clicks Save twice for the same analysis payload.
    db.query(SimulationWaferRecord).filter(SimulationWaferRecord.session_id == response.session_id).delete(synchronize_session=False)
    existing = db.query(SimulationSession).filter(SimulationSession.session_id == response.session_id).one_or_none()
    if existing is not None:
        db.delete(existing)
        db.flush()

    max_wafers = max(0, int(settings.simulator_max_persist_wafers))
    matrix_size = max(16, min(160, int(settings.simulator_persist_matrix_size)))
    method = settings.simulator_persist_downsample_method if settings.simulator_persist_downsample_method in {"nearest", "area"} else "area"
    persisted_wafers = response.wafers[:max_wafers]
    persist_matrices = bool(getattr(settings, "simulator_persist_matrices", False))
    db.add(
        SimulationSession(
            session_id=response.session_id,
            created_at=response.created_at,
            scenario_name=response.params.scenario_name,
            wafer_count=response.summary.total_wafers,
            persisted_wafer_count=len(persisted_wafers),
            matrix_persist_size=matrix_size,
            response_payload_kind="summary_plus_seeded_rows" if not persist_matrices else "summary_plus_compact_matrices",
            params_json=json.dumps(response.params.model_dump(mode="json"), ensure_ascii=False),
            summary_json=json.dumps(response.summary.model_dump(mode="json"), ensure_ascii=False),
            model_json=json.dumps(response.model.model_dump(mode="json"), ensure_ascii=False),
            response_json=None,
        )
    )
    for wafer in persisted_wafers:
        wafer_payload = wafer.model_dump(mode="json", exclude={"matrix"})
        wafer_payload["_persist_downsample_method"] = method
        matrix_json = None
        if persist_matrices:
            matrix = downsample_matrix(np.asarray(wafer.matrix), matrix_size, method=method).astype(int).tolist()
            matrix_json = json.dumps(matrix, ensure_ascii=False)
        db.add(
            SimulationWaferRecord(
                session_id=response.session_id,
                wafer_id=wafer.id,
                wafer_index=wafer.wafer_index,
                lot_id=wafer.lot_id,
                tool_id=wafer.tool_id,
                chamber_id=wafer.chamber_id,
                process_step=wafer.process_step,
                true_label=wafer.true_label,
                predicted_label=wafer.predicted_label,
                confidence=wafer.confidence,
                risk_score=wafer.risk_score,
                severity=wafer.severity,
                wafer_json=json.dumps(wafer_payload, ensure_ascii=False),
                matrix_json=matrix_json,
            )
        )
    db.commit()
    storage_note = f"{matrix_size}px {method} matrices" if bool(getattr(settings, "simulator_persist_matrices", False)) else f"matrix seeds; UI matrices regenerate at {matrix_size}px"
    if len(persisted_wafers) < len(response.wafers):
        note = f"Persisted first {len(persisted_wafers)} of {len(response.wafers)} wafer rows with {storage_note}; summary keeps full-run totals."
    else:
        note = f"Persisted {len(persisted_wafers)} wafer rows with {storage_note}."
    return len(persisted_wafers), note


def _rehydrate_session(row: SimulationSession, db: Session) -> SimulatorResponse | None:
    try:
        params = SimulatorRequest.model_validate(json.loads(row.params_json))
        summary = SimulatorSummary.model_validate(json.loads(row.summary_json))
        model_payload = json.loads(row.model_json) if getattr(row, "model_json", None) else {
            "loaded": False,
            "model_version": "stored-session",
            "checkpoint_path": "unknown",
            "class_names": LABELS,
            "model_kind": "stored",
        }
        model = ModelMetadata.model_validate(model_payload)
    except Exception:
        return None

    wafer_rows = db.query(SimulationWaferRecord).filter(SimulationWaferRecord.session_id == row.session_id).order_by(SimulationWaferRecord.wafer_index).all()
    wafers: list[SimulatorWafer] = []
    for wafer_row in wafer_rows:
        payload = json.loads(wafer_row.wafer_json)
        if wafer_row.matrix_json:
            payload["matrix"] = json.loads(wafer_row.matrix_json)
        else:
            payload["matrix"] = _regenerate_persisted_matrix(
                params,
                payload,
                matrix_size=getattr(row, "matrix_persist_size", None) or 40,
                method=payload.get("_persist_downsample_method", "area"),
            )
        wafers.append(SimulatorWafer.model_validate(payload))
    persisted = getattr(row, "persisted_wafer_count", len(wafers)) or len(wafers)
    note = f"Loaded saved wafer subset: {persisted} of {row.wafer_count} wafers. Full aggregate summary is preserved."
    return SimulatorResponse(
        session_id=row.session_id,
        created_at=row.created_at,
        params=params,
        wafers=wafers,
        summary=summary,
        model=model,
        persisted_wafer_count=persisted,
        persistence_note=note,
    )


def _regenerate_persisted_matrix(params: SimulatorRequest, wafer_payload: dict[str, Any], matrix_size: int, method: str = "area") -> list[list[int]]:
    """Regenerate a compact UI matrix from the stored deterministic wafer seed.

    Saved sessions intentionally avoid matrix_json by default. This keeps SQLite
    rows small while still allowing the cockpit to re-open trellis/detail views.
    """
    seed = wafer_payload.get("matrix_seed")
    if seed is None:
        return []
    matrix = generate_wafer_map(
        str(wafer_payload.get("true_label", "Random")),
        size=params.size,
        noise_level=params.noise_level,
        defect_density_scale=params.defect_density_scale,
        rng=np.random.default_rng(int(seed)),
        secondary_label=wafer_payload.get("secondary_label"),
    )
    safe_method = method if method in {"nearest", "area"} else "area"
    return downsample_matrix(matrix, max(16, min(160, int(matrix_size))), method=safe_method).astype(int).tolist()


def _csv_response(headers: list[str], rows: list[dict[str, Any]], filename: str) -> StreamingResponse:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except Exception:
        return None
