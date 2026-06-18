from __future__ import annotations

from fastapi import APIRouter, Depends

from wafer_vision_api.dependencies import get_app_settings, get_model_service
from wafer_vision_api.redis_client import ping_redis
from wafer_vision_api.schemas import HealthResponse, ModelMetadata
from wafer_vision_api.services.model_service import WaferModelService
from wafer_vision_api.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    model_service: WaferModelService = Depends(get_model_service),
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    redis_ok = ping_redis(settings)
    redis_required = settings.normalized_job_backend in {"rq", "celery"} or settings.normalized_rate_limit_backend == "redis"
    status = "ok"
    if not model_service.loaded:
        status = "degraded"
    if redis_required and not redis_ok:
        status = "degraded"
    return HealthResponse(
        status=status,
        app=settings.app_name,
        environment=settings.environment,
        model_loaded=model_service.loaded,
        database=settings.database_url,
        redis="ok" if redis_ok else "unavailable",
        job_backend=settings.normalized_job_backend,
    )


@router.get("/model", response_model=ModelMetadata, tags=["model"])
def model_info(model_service: WaferModelService = Depends(get_model_service)) -> ModelMetadata:
    return model_service.metadata()
