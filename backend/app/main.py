"""Brain Tumor AI application entry point.

Assembles the FastAPI application, security middleware, exception mapping and
versioned routers. The ML model is warmed once at startup and cached.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import admin, health, history, models as models_api, predictions, reports
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.security import SecurityHeadersMiddleware
from app.services.errors import AppError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    # Database connectivity + registry sync (best effort at startup).
    try:
        from app.db.database import init_db

        init_db()
        logger.info("Database initialized and model registry synced.")
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Database initialization failed: %s", exc)

    # Warm up the active model so the first prediction is fast.
    try:
        from app.db.database import SessionLocal
        from app.services.model_manager import ModelManager

        db = SessionLocal()
        try:
            record = ModelManager().get_active(db)
            ModelManager().get_backend(record)
            logger.info(
                "Active model warmed: %s v%s",
                record.name,
                record.version,
            )
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Model warm-up failed: %s", exc)

    yield

    from app.services.model_manager import ModelManager

    ModelManager().shutdown()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-assisted analysis of brain MRI images for research and educational "
        "purposes. Results are not medical diagnoses."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail if hasattr(exc, "detail") else exc.message},
    )


@app.get("/", tags=["meta"], include_in_schema=False)
async def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "api": settings.api_v1_prefix,
    }


for router in (
    health.router,
    predictions.router,
    history.router,
    models_api.router,
    reports.router,
    admin.router,
):
    app.include_router(router, prefix=settings.api_v1_prefix)


__all__ = ["app"]