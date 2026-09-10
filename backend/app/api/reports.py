"""Report endpoints: generate, metadata, download."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.deps import report_rate_limit
from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import ReportRecord
from app.repositories.predictions import AnalysisRepository
from app.schemas.report import ReportResponse
from app.services.errors import NotFoundError, ReportGenerationError
from app.services.reporting import ReportGenerator

router = APIRouter(tags=["reports"])


class ReportRequest(BaseModel):
    analysis_id: str


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