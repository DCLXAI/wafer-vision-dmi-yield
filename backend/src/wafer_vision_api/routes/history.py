from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from wafer_vision_api.database import get_db
from wafer_vision_api.schemas import HistoryPage, PredictionDetail, StatsSummary
from wafer_vision_api.services.history_service import get_prediction_detail, list_predictions, stats_summary

router = APIRouter(tags=["history"])


@router.get("/predictions", response_model=HistoryPage)
def prediction_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    label: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> HistoryPage:
    return list_predictions(db, limit=limit, offset=offset, label=label)


@router.get("/predictions/{prediction_id}", response_model=PredictionDetail)
def prediction_detail(prediction_id: int, db: Session = Depends(get_db)) -> PredictionDetail:
    detail = get_prediction_detail(db, prediction_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction record not found.")
    return detail


@router.get("/stats/summary", response_model=StatsSummary)
def summary(db: Session = Depends(get_db)) -> StatsSummary:
    return stats_summary(db)
