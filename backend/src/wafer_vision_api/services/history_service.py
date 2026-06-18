from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from wafer_vision_api.db_models import PredictionRecord
from wafer_vision_api.schemas import (
    HistoryPage,
    LabelCount,
    PredictionDetail,
    PredictionHistoryItem,
    StatsSummary,
    TopKPrediction,
)


def create_prediction_record(
    db: Session,
    *,
    filename: str | None,
    content_type: str | None,
    input_kind: str,
    rows: int | None,
    cols: int | None,
    min_value: float | None,
    max_value: float | None,
    predicted_label: str,
    confidence: float,
    top_k: list[TopKPrediction],
    model_version: str,
    checkpoint_path: str,
    inference_ms: float,
    client_note: str | None = None,
) -> PredictionRecord:
    record = PredictionRecord(
        filename=filename,
        content_type=content_type,
        input_kind=input_kind,
        rows=rows,
        cols=cols,
        min_value=min_value,
        max_value=max_value,
        predicted_label=predicted_label,
        confidence=confidence,
        top_k_json=json.dumps([_dump_model(item) for item in top_k], ensure_ascii=False),
        model_version=model_version,
        checkpoint_path=checkpoint_path,
        inference_ms=inference_ms,
        client_note=client_note,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_predictions(db: Session, *, limit: int = 50, offset: int = 0, label: str | None = None) -> HistoryPage:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    base: Select[tuple[PredictionRecord]] = select(PredictionRecord)
    count_stmt = select(func.count(PredictionRecord.id))
    if label:
        base = base.where(PredictionRecord.predicted_label == label)
        count_stmt = count_stmt.where(PredictionRecord.predicted_label == label)

    total = int(db.execute(count_stmt).scalar_one())
    rows = db.execute(
        base.order_by(PredictionRecord.created_at.desc()).limit(limit).offset(offset)
    ).scalars().all()
    return HistoryPage(
        total=total,
        limit=limit,
        offset=offset,
        items=[_history_item(row) for row in rows],
    )


def get_prediction_detail(db: Session, prediction_id: int) -> PredictionDetail | None:
    record = db.get(PredictionRecord, prediction_id)
    if record is None:
        return None
    return PredictionDetail(
        id=record.id,
        created_at=_ensure_datetime(record.created_at),
        filename=record.filename,
        content_type=record.content_type,
        input_kind=record.input_kind,
        rows=record.rows,
        cols=record.cols,
        min_value=record.min_value,
        max_value=record.max_value,
        predicted_label=record.predicted_label,
        confidence=record.confidence,
        top_k=_parse_top_k(record.top_k_json),
        model_version=record.model_version,
        checkpoint_path=record.checkpoint_path,
        inference_ms=record.inference_ms,
        client_note=record.client_note,
    )


def stats_summary(db: Session, *, latest_limit: int = 10) -> StatsSummary:
    total = int(db.execute(select(func.count(PredictionRecord.id))).scalar_one())
    avg_conf = db.execute(select(func.avg(PredictionRecord.confidence))).scalar_one()
    label_rows = db.execute(
        select(PredictionRecord.predicted_label, func.count(PredictionRecord.id))
        .group_by(PredictionRecord.predicted_label)
        .order_by(func.count(PredictionRecord.id).desc())
    ).all()
    latest = db.execute(
        select(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc())
        .limit(max(1, min(latest_limit, 50)))
    ).scalars().all()

    return StatsSummary(
        total_predictions=total,
        average_confidence=float(avg_conf) if avg_conf is not None else None,
        label_counts=[LabelCount(label=str(label), count=int(count)) for label, count in label_rows],
        latest=[_history_item(row) for row in latest],
    )


def _history_item(record: PredictionRecord) -> PredictionHistoryItem:
    return PredictionHistoryItem(
        id=record.id,
        created_at=_ensure_datetime(record.created_at),
        filename=record.filename,
        input_kind=record.input_kind,
        predicted_label=record.predicted_label,
        confidence=record.confidence,
        inference_ms=record.inference_ms,
        rows=record.rows,
        cols=record.cols,
        model_version=record.model_version,
    )


def _dump_model(item: TopKPrediction) -> dict:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return item.dict()


def _parse_top_k(payload: str) -> list[TopKPrediction]:
    try:
        raw = json.loads(payload)
        return [TopKPrediction(**item) for item in raw]
    except Exception:
        return []


def _ensure_datetime(value: datetime) -> datetime:
    return value
