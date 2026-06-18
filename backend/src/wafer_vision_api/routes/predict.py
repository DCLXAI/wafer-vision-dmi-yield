from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from wafer_vision_api.database import get_db
from wafer_vision_api.dependencies import get_model_service
from wafer_vision_api.input_loader import InputDecodeError, describe_array, load_wafer_array_from_inline, load_wafer_array_from_upload
from wafer_vision_api.schemas import ArrayPredictRequest, InputMetadata, PredictResponse
from wafer_vision_api.services.history_service import create_prediction_record
from wafer_vision_api.services.model_service import ModelNotLoadedError, WaferModelService
from wafer_vision_api.settings import get_settings

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictResponse, status_code=status.HTTP_201_CREATED)
async def predict_upload(
    file: UploadFile = File(...),
    note: str | None = Form(default=None),
    db: Session = Depends(get_db),
    model_service: WaferModelService = Depends(get_model_service),
) -> PredictResponse:
    settings = get_settings()
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max upload size is {settings.max_upload_mb} MB.",
        )

    try:
        wafer = load_wafer_array_from_upload(content, file.filename, file.content_type)
    except InputDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return _predict_and_store(
        wafer=wafer,
        filename=file.filename,
        content_type=file.content_type,
        input_kind="upload",
        note=note,
        db=db,
        model_service=model_service,
    )


@router.post("/array", response_model=PredictResponse, status_code=status.HTTP_201_CREATED)
def predict_array(
    payload: ArrayPredictRequest,
    db: Session = Depends(get_db),
    model_service: WaferModelService = Depends(get_model_service),
) -> PredictResponse:
    try:
        wafer = load_wafer_array_from_inline(payload.wafer_map)
    except InputDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return _predict_and_store(
        wafer=wafer,
        filename=payload.filename,
        content_type="application/json",
        input_kind="inline-array",
        note=payload.note,
        db=db,
        model_service=model_service,
    )


def _predict_and_store(
    *,
    wafer,
    filename: str | None,
    content_type: str | None,
    input_kind: str,
    note: str | None,
    db: Session,
    model_service: WaferModelService,
) -> PredictResponse:
    try:
        prediction = model_service.predict(wafer)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    meta = describe_array(wafer)
    model_meta = model_service.metadata()
    record = create_prediction_record(
        db,
        filename=filename,
        content_type=content_type,
        input_kind=input_kind,
        rows=meta["rows"],
        cols=meta["cols"],
        min_value=meta["min_value"],
        max_value=meta["max_value"],
        predicted_label=prediction.label,
        confidence=prediction.confidence,
        top_k=prediction.top_k,
        model_version=model_meta.model_version,
        checkpoint_path=model_meta.checkpoint_path,
        inference_ms=prediction.inference_ms,
        client_note=note,
    )

    return PredictResponse(
        id=record.id,
        label=prediction.label,
        confidence=prediction.confidence,
        top_k=prediction.top_k,
        inference_ms=prediction.inference_ms,
        created_at=record.created_at,
        input=InputMetadata(
            filename=filename,
            content_type=content_type,
            input_kind=input_kind,
            rows=meta["rows"],
            cols=meta["cols"],
            min_value=meta["min_value"],
            max_value=meta["max_value"],
        ),
        model=model_meta,
    )
