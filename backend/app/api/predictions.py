"""Prediction endpoints: upload validation, inference, retrieval, media."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import prediction_rate_limit
from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import AnalysisRecord
from app.repositories.predictions import AnalysisRepository
from app.schemas.prediction import (
    ModelMeta,
    PredictionDetailResponse,
    PredictionLinks,
    PredictionResponse,
    ReportInfo,
)
from app.services.errors import NotFoundError
from app.services.explainability import render_attention_overlay
from app.services.inference import InferenceService
from app.services.validation import validate_upload

router = APIRouter(tags=["predictions"])

_MAX_READ = settings.max_upload_size_mb * 1024 * 1024


def _safe_name(filename: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return name[:120]


def _build_links(analysis_id: str) -> PredictionLinks:
    return PredictionLinks(
        image=f"/api/v1/predictions/{analysis_id}/image",
        gradcam=f"/api/v1/predictions/{analysis_id}/gradcam",
    )


def _to_response(db, model) -> PredictionDetailResponse:
    probabilities = json.loads(model.probabilities_json or "{}")
    model_meta = ModelMeta(
        name=model.model_name,
        version=model.model_version,
        architecture="",
        input_size=0,
    )
    # Fetch full model metadata from the registry-backed store.
    from app.repositories.models import ModelRepository

    record = ModelRepository(db).get_by_name_version(model.model_name, model.model_version)
    if record is not None:
        model_meta = ModelMeta(
            name=record.name,
            version=record.version,
            architecture=record.architecture,
            input_size=record.input_size,
            dataset_version=record.dataset_version,
        )

    return PredictionDetailResponse(
        analysis_id=model.id,
        prediction=model.prediction_label,
        label=model.prediction_label.replace("_", " ").title(),
        confidence=model.confidence,
        low_confidence=model.low_confidence,
        probabilities=probabilities,
        model=model_meta,
        processing_time_ms=model.processing_time_ms,
        created_at=model.created_at.isoformat() + "Z",
        report_available=model.report is not None,
        filename=model.filename,
        warnings=[],
        note=(
            "This AI-generated result is intended for research and educational "
            "assistance only and does not constitute a medical diagnosis."
        ),
        report=(
            ReportInfo(id=model.report.id, available=True)
            if model.report is not None
            else None
        ),
        links=_build_links(model.id),
    )


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    status_code=201,
    summary="Analyze an uploaded brain MRI",
    dependencies=[prediction_rate_limit],
)
async def create_prediction(
    file: UploadFile = File(...),
) -> PredictionResponse:
    data = await file.read()
    return _run_analysis(data, file.filename or "upload.png")


def _run_analysis(data: bytes, filename: str) -> PredictionResponse:
    """Heavy analysis work runs on a worker thread (sync, not awaited)."""
    validated = validate_upload(filename, data)

    db = SessionLocal()
    try:
        result = InferenceService().analyze(validated.bytes, db)

        analysis = AnalysisRecord(
            filename=filename,
            image_path="",
            grad_cam_path="",
            prediction_label=result.prediction,
            class_index=result.class_index,
            confidence=result.confidence,
            low_confidence=result.low_confidence,
            probabilities_json=json.dumps(result.probabilities),
            model_name=result.model["name"],
            model_version=result.model["version"],
            processing_time_ms=result.processing_time_ms,
            image_checksum=validated.checksum,
        )
        repo = AnalysisRepository(db)
        stored = repo.create(analysis)

        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        image_path = upload_dir / f"{stored.id}_{_safe_name(stored.filename)}"
        image_path.write_bytes(validated.bytes)

        grad_dir = Path(settings.gradcam_dir)
        grad_dir.mkdir(parents=True, exist_ok=True)
        overlay = render_attention_overlay(result.prepared.array_rgb, result.heatmap)
        overlay_path = grad_dir / f"{stored.id}_overlay.png"
        overlay_path.write_bytes(overlay)

        stored.image_path = str(image_path)
        stored.grad_cam_path = str(overlay_path)
        db.add(stored)
        db.commit()
        db.refresh(stored)

        return _to_response(db, stored)
    finally:
        db.close()


@router.get(
    "/predictions/{analysis_id}",
    response_model=PredictionDetailResponse,
    summary="Get a stored analysis",
)
def get_prediction(analysis_id: str) -> PredictionDetailResponse:
    db = SessionLocal()
    try:
        repo = AnalysisRepository(db)
        model = repo.get(analysis_id)
        if model is None:
            raise NotFoundError("Analysis not found.")
        return _to_response(db, model)
    finally:
        db.close()


@router.get(
    "/predictions/{analysis_id}/image",
    summary="Download the uploaded MRI file",
    response_class=FileResponse,
)
def get_analysis_image(analysis_id: str) -> FileResponse:
    db = SessionLocal()
    try:
        model = AnalysisRepository(db).get(analysis_id)
        if model is None or not model.image_path:
            raise NotFoundError("Analysis not found.")
    finally:
        db.close()
    path = Path(model.image_path)
    if not path.exists():
        raise NotFoundError("Image file is missing.")
    return FileResponse(str(path), media_type="image/png", filename=model.filename)


@router.get(
    "/predictions/{analysis_id}/gradcam",
    summary="Download the AI attention visualization",
    response_class=FileResponse,
)
def get_analysis_gradcam(analysis_id: str) -> FileResponse:
    db = SessionLocal()
    try:
        model = AnalysisRepository(db).get(analysis_id)
        if model is None or not model.grad_cam_path:
            raise NotFoundError("Attention visualization is not available.")
    finally:
        db.close()
    path = Path(model.grad_cam_path)
    if not path.exists():
        raise NotFoundError("Attention visualization file is missing.")
    return FileResponse(str(path), media_type="image/png")