from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import AnalysisRecord, ReportRecord


class AnalysisRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, analysis: AnalysisRecord) -> AnalysisRecord:
        self.db.add(analysis)
        self.db.commit()
        self.db.refresh(analysis)
        return analysis

    def get(self, analysis_id: str) -> AnalysisRecord | None:
        return (
            self.db.query(AnalysisRecord)
            .filter(AnalysisRecord.id == analysis_id)
            .first()
        )

    def find_by_checksum(self, checksum: str) -> AnalysisRecord | None:
        return (
            self.db.query(AnalysisRecord)
            .filter(AnalysisRecord.image_checksum == checksum)
            .first()
        )

    def attach_report(self, analysis_id: str, report: ReportRecord) -> AnalysisRecord | None:
        analysis = self.get(analysis_id)
        if analysis is None:
            return None
        analysis.report = report
        self.db.commit()
        self.db.refresh(analysis)
        return analysis