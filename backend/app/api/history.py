"""Analysis history endpoints with pagination, search and filtering."""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Query

from app.db.database import SessionLocal
from app.repositories.history import HistoryRepository
from app.repositories.predictions import AnalysisRepository
from app.schemas.history import HistoryItem, HistoryListResponse
from app.services.errors import NotFoundError

router = APIRouter(tags=["history"])


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(
            tzinfo=None
        )
    except ValueError:
        return None


@router.get(
    "/history",
    response_model=HistoryListResponse,
    summary="List past analyses",
)
def list_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str | None = Query(None),
    prediction: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
) -> HistoryListResponse:
    db = SessionLocal()
    try:
        repo = HistoryRepository(db)
        items, total = repo.list(
            page=page,
            page_size=page_size,
            search=search,
            prediction=prediction,
            date_from=_parse_date(date_from),
            date_to=_parse_date(date_to),
        )
        history_items = [
            HistoryItem(
                id=item.id,
                created_at=item.created_at.isoformat() + "Z",
                filename=item.filename,
                prediction=item.prediction_label,
                confidence=item.confidence,
                probabilities=json.loads(item.probabilities_json or "{}") or {},
                model_name=item.model_name,
                model_version=item.model_version,
                low_confidence=item.low_confidence,
                processing_time_ms=item.processing_time_ms,
                report_available=item.report is not None,
            )
            for item in items
        ]
        pages = max(1, (total + page_size - 1) // page_size)
        return HistoryListResponse(
            items=history_items,
            total=total,
            page=page,
            page_size=page_size,
            pages=pages,
        )
    finally:
        db.close()


@router.get(
    "/history/{analysis_id}",
    response_model=HistoryItem,
    summary="Get a single history item",
)
def get_history_item(analysis_id: str) -> HistoryItem:
    db = SessionLocal()
    try:
        item = AnalysisRepository(db).get(analysis_id)
        if item is None:
            raise NotFoundError("Analysis not found.")
        return HistoryItem(
            id=item.id,
            created_at=item.created_at.isoformat() + "Z",
            filename=item.filename,
            prediction=item.prediction_label,
            confidence=item.confidence,
            probabilities=json.loads(item.probabilities_json or "{}"),
            model_name=item.model_name,
            model_version=item.model_version,
            low_confidence=item.low_confidence,
            processing_time_ms=item.processing_time_ms,
            report_available=item.report is not None,
        )
    finally:
        db.close()