"""Report endpoints: list, generate, metadata, download."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.deps import report_rate_limit
from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import AnalysisRecord, ReportRecord
from app.repositories.predictions import AnalysisRepository
from app.schemas.report import ReportResponse
from app.services.errors import NotFoundError, ReportGenerationError
from app.services.reporting import ReportGenerator

router = APIRouter(tags=["reports"])


class ReportRequest(BaseModel):
    analysis_id: str


class ReportListItem(BaseModel):
    id: str
    analysis_id: str
    created_at: str
    size_bytes: int
    filename: str | None
    prediction: str | None
    confidence: float | None
    model_name: str | None
    model_version: str | None
    download_url: str


class ReportListResponse(BaseModel):
    items: list[ReportListItem]
    total: int
    page: int
    page_size: int
    pages: int


@router.get(
    "/reports",
    response_model=ReportListResponse,
    summary="List generated reports",
)
def list_reports(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> ReportListResponse:
    db = SessionLocal()
    try:
        query = (
            db.query(ReportRecord)
            .order_by(ReportRecord.generated_at.desc())
        )
        total = query.count()
        reports = (
            query.offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        items: list[ReportListItem] = []
        for report in reports:
            analysis: AnalysisRecord | None = None
            if report.analysis_id:
                analysis = db.get(AnalysisRecord, report.analysis_id)
            probabilities = "{}"
            if analysis is not None:
                probabilities = analysis.probabilities_json or "{}"
            confidences = json.loads(probabilities) if probabilities else {}
            confidence = (
                float(confidences.get(analysis.prediction_label) or 0.0)
                if analysis is not None and confidences
                else None
            )
            items.append(
                ReportListItem(
                    id=report.id,
                    analysis_id=report.analysis_id,
                    created_at=report.generated_at.isoformat() + "Z",
                    size_bytes=report.size_bytes,
                    filename=analysis.filename if analysis else None,
                    prediction=analysis.prediction_label if analysis else None,
                    confidence=confidence,
                    model_name=analysis.model_name if analysis else None,
                    model_version=analysis.model_version if analysis else None,
                    download_url=f"/api/v1/reports/{report.id}/download",
                )
            )
        pages = max(1, (total + page_size - 1) // page_size)
        return ReportListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            pages=pages,
        )
    finally:
        db.close()


@router.post(
    "/reports",
    response_model=ReportResponse,
    status_code=201,
    summary="Generate an AI-Assisted Brain MRI Analysis Report",
    dependencies=[report_rate_limit],
)
def generate_report(body: ReportRequest) -> ReportResponse:
    db = SessionLocal()
    try:
        analysis = AnalysisRepository(db).get(body.analysis_id)
        if analysis is None:
            raise NotFoundError("Analysis not found.")

        if analysis.report is not None:
            report = analysis.report
            return ReportResponse(
                id=report.id,
                analysis_id=report.analysis_id,
                size_bytes=report.size_bytes,
                generated_at=report.generated_at.isoformat() + "Z",
                download_url=f"/api/v1/reports/{report.id}/download",
            )

        overlay_bytes = b""
        if analysis.grad_cam_path and Path(analysis.grad_cam_path).exists():
            overlay_bytes = Path(analysis.grad_cam_path).read_bytes()

        generator = ReportGenerator()
        report_path = generator.generate(analysis, overlay_bytes)

        report = ReportRecord(
            analysis_id=analysis.id,
            file_path=str(report_path),
            size_bytes=report_path.stat().st_size,
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        analysis.report = report
        db.commit()

        return ReportResponse(
            id=report.id,
            analysis_id=report.analysis_id,
            size_bytes=report.size_bytes,
            generated_at=report.generated_at.isoformat() + "Z",
            download_url=f"/api/v1/reports/{report.id}/download",
        )
    finally:
        db.close()


@router.get(
    "/reports/{report_id}",
    response_model=ReportResponse,
    summary="Get report metadata",
)
def get_report(report_id: str) -> ReportResponse:
    db = SessionLocal()
    try:
        report = (
            db.query(ReportRecord).filter(ReportRecord.id == report_id).first()
        )
        if report is None:
            raise NotFoundError("Report not found.")
        return ReportResponse(
            id=report.id,
            analysis_id=report.analysis_id,
            size_bytes=report.size_bytes,
            generated_at=report.generated_at.isoformat() + "Z",
            download_url=f"/api/v1/reports/{report.id}/download",
        )
    finally:
        db.close()


@router.get(
    "/reports/{report_id}/download",
    summary="Download the report PDF",
    response_class=FileResponse,
)
def download_report(report_id: str) -> FileResponse:
    db = SessionLocal()
    try:
        report = (
            db.query(ReportRecord).filter(ReportRecord.id == report_id).first()
        )
        if report is None:
            raise NotFoundError("Report not found.")
        path = Path(report.file_path)
        if not path.exists():
            raise ReportGenerationError("Report file is missing from disk.")
        return FileResponse(
            str(path),
            media_type="application/pdf",
            filename=f"brain-mri-analysis-{report.analysis_id}.pdf",
        )
    finally:
        db.close()