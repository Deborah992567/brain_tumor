"""Model management endpoints: listing, detail, activation, reload."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends

from app.core.security import require_admin_key
from app.db.database import SessionLocal
from app.repositories.models import ModelRepository
from app.schemas.model import ModelInfo, ModelListResponse
from app.services.errors import NotFoundError
from app.services.model_manager import ModelManager

router = APIRouter(tags=["models"])


def _to_info(record) -> ModelInfo:
    return ModelInfo(
        id=record.id,
        name=record.name,
        version=record.version,
        architecture=record.architecture,
        file_path=record.file_path,
        input_size=record.input_size,
        dataset_version=record.dataset_version,
        training_datetime=(
            record.training_datetime.isoformat() + "Z"
            if record.training_datetime
            else None
        ),
        metrics=json.loads(record.metrics_json or "{}"),
        status=record.status,
        description=record.description,
        is_active=record.is_active,
        created_at=record.created_at.isoformat() + "Z",
    )


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List registered models and the active model",
)
def list_models() -> ModelListResponse:
    db = SessionLocal()
    try:
        manager = ModelManager()
        records = manager.list(db)
        active = manager.get_active(db)
        return ModelListResponse(
            models=[_to_info(r) for r in records],
            active=_to_info(active) if active else None,
        )
    finally:
        db.close()


@router.get(
    "/models/{model_id}",
    response_model=ModelInfo,
    summary="Get a single model version",
)
def get_model(model_id: int) -> ModelInfo:
    db = SessionLocal()
    try:
        record = ModelRepository(db).get(model_id)
        if record is None:
            raise NotFoundError("Model not found.")
        return _to_info(record)
    finally:
        db.close()


@router.post(
    "/models/{model_id}/activate",
    response_model=ModelInfo,
    summary="Activate a model version (admin)",
    dependencies=[Depends(require_admin_key)],
)
def activate_model(model_id: int) -> ModelInfo:
    db = SessionLocal()
    try:
        record = ModelManager().set_active(db, model_id)
        ModelManager().reload()
        return _to_info(record)
    finally:
        db.close()


@router.post(
    "/models/reload",
    response_model=dict,
    summary="Reload all model backends (admin)",
    dependencies=[Depends(require_admin_key)],
)
def reload_models() -> dict:
    ModelManager().reload()
    return {"status": "reloaded"}