from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from wafer_vision_api.database import init_db
from wafer_vision_api.middleware import ApiKeyAndRateLimitMiddleware
from wafer_vision_api.routes import features, health, history, predict, simulator
from wafer_vision_api.services.model_service import WaferModelService
from wafer_vision_api.settings import Settings, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = getattr(app.state, "settings", None) or get_settings()
    init_db(settings.database_url)
    model_service = WaferModelService(
        checkpoint_path=settings.checkpoint_path,
        device=settings.device,
        top_k=settings.top_k,
        model_kind=settings.model_kind,
    )
    model_service.load()
    app.state.model_service = model_service
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.9.0",
        description="FastAPI backend for WaferVision DMI yield intelligence: rare-defect classification, process feature extraction, wafer-lot simulation, chamber traceback, and Postgres + Redis/RQ job infrastructure.",
        lifespan=lifespan,
    )

    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(ApiKeyAndRateLimitMiddleware, settings=settings)

    # Health is intentionally exposed at both `/health` and `/api/v1/health`:
    # root-level uptime checks stay simple while the versioned API remains tidy.
    app.include_router(health.router)
    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(predict.router, prefix=settings.api_prefix)
    app.include_router(features.router, prefix=settings.api_prefix)
    app.include_router(history.router, prefix=settings.api_prefix)
    app.include_router(simulator.router, prefix=settings.api_prefix)
    return app


app = create_app()
