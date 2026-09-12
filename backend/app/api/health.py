"""Health endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from app.api.deps import health_rate_limit
from app.core.config import settings
from app.db.database import engine
from app.schemas.health import HealthResponse
from app.services.model_manager import ModelManager

router = APIRouter(tags=["health"])

_STARTED_AT = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    dependencies=[health_rate_limit],
)
def health() -> HealthResponse:
    db_status = "ok"
    db_detail = "connected"
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
    except Exception:
        db_status = "error"
        db_detail = "unavailable"

    model_info: dict | None = None
    try:
        from app.db.database import SessionLocal

        db = SessionLocal()
        try:
            active = ModelManager().get_active(db)
            model_info = {
                "name": active.name,
                "version": active.version,
                "architecture": active.architecture,
                "status": active.status,
            }
        finally:
            db.close()
    except Exception:
        model_info = None

    return HealthResponse(
        status="ok" if db_status == "ok" and model_info is not None else "degraded",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        uptime_seconds=round(time.time() - _STARTED_AT, 1),
        database=db_detail,
        model=model_info,
        admin_enabled=settings.admin_enabled,
    )