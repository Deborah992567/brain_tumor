from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.db.models import AnalysisRecord, ReportRecord


class HistoryRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        search: str | None = None,
        prediction: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> tuple[list[AnalysisRecord], int]:
        query = self.db.query(AnalysisRecord)

        if prediction:
            query = query.filter(AnalysisRecord.prediction_label == prediction)

        if date_from:
            query = query.filter(AnalysisRecord.created_at >= date_from)
        if date_to:
            query = query.filter(AnalysisRecord.created_at <= date_to)

        if search:
            like = f"%{search}%"
            query = query.filter(
                or_(
                    AnalysisRecord.filename.ilike(like),
                    AnalysisRecord.id.ilike(like),
                    AnalysisRecord.model_name.ilike(like),
                )
            )

        total = query.count()
        items = (
            query.order_by(AnalysisRecord.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return list(items), total

    def prediction_distribution(self) -> dict[str, int]:
        rows = (
            self.db.query(AnalysisRecord.prediction_label, func.count(AnalysisRecord.id))
            .group_by(AnalysisRecord.prediction_label)
            .all()
        )
        return {label: count for label, count in rows}

    def counts(self) -> dict:
        return {
            "total_analyses": self.db.query(AnalysisRecord).count(),
            "total_reports": self.db.query(ReportRecord).count(),
        }

    def recent(self, limit: int = 6) -> list[AnalysisRecord]:
        return (
            self.db.query(AnalysisRecord)
            .order_by(AnalysisRecord.created_at.desc())
            .limit(limit)
            .all()
        )