from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from wafer_vision.features import (
    FEATURE_SCHEMA,
    GEOMETRY_FEATURE_NAMES,
    RADON_MEAN_FEATURE_NAMES,
    RADON_STD_FEATURE_NAMES,
    REGION_FEATURE_NAMES,
    extract_feature_groups,
)
from wafer_vision_api.input_loader import InputDecodeError, describe_array, load_wafer_array_from_inline, load_wafer_array_from_upload
from wafer_vision_api.schemas import ArrayPredictRequest, FeatureGroup, FeatureResponse, InputMetadata
from wafer_vision_api.settings import get_settings

router = APIRouter(prefix="/features", tags=["kaggle-features"])


@router.post("", response_model=FeatureResponse, status_code=status.HTTP_200_OK)
async def extract_features_upload(file: UploadFile = File(...), note: str | None = Form(default=None)) -> FeatureResponse:
    settings = get_settings()
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"File too large. Max upload size is {settings.max_upload_mb} MB.")
    try:
        wafer = load_wafer_array_from_upload(content, file.filename, file.content_type)
    except InputDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _feature_response(wafer, filename=file.filename, content_type=file.content_type, input_kind="upload")


@router.post("/array", response_model=FeatureResponse, status_code=status.HTTP_200_OK)
def extract_features_array(payload: ArrayPredictRequest) -> FeatureResponse:
    try:
        wafer = load_wafer_array_from_inline(payload.wafer_map)
    except InputDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _feature_response(wafer, filename=payload.filename, content_type="application/json", input_kind="inline-array")


def _feature_response(wafer, filename: str | None, content_type: str | None, input_kind: str) -> FeatureResponse:
    try:
        groups = extract_feature_groups(wafer)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Feature extraction failed: {exc}") from exc
    meta = describe_array(wafer)
    return FeatureResponse(
        feature_dim=len(FEATURE_SCHEMA),
        feature_schema=FEATURE_SCHEMA,
        vector=groups.vector,
        named_vector=groups.named_vector,
        groups=[
            FeatureGroup(name="region_density", values=groups.region_density, labels=REGION_FEATURE_NAMES),
            FeatureGroup(name="radon_mean", values=groups.radon_mean, labels=RADON_MEAN_FEATURE_NAMES),
            FeatureGroup(name="radon_std", values=groups.radon_std, labels=RADON_STD_FEATURE_NAMES),
            FeatureGroup(name="geometry", values=groups.geometry, labels=GEOMETRY_FEATURE_NAMES),
        ],
        input=InputMetadata(filename=filename, content_type=content_type, input_kind=input_kind, rows=meta["rows"], cols=meta["cols"], min_value=meta["min_value"], max_value=meta["max_value"]),
    )
