from __future__ import annotations

from fastapi import Request

from wafer_vision_api.services.model_service import WaferModelService
from wafer_vision_api.settings import Settings


def get_model_service(request: Request) -> WaferModelService:
    return request.app.state.model_service


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings
