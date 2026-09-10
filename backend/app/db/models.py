"""SQLAlchemy ORM models for analysis history, model versions and reports.

No personal or medical-identifying information is stored. Each analysis
record only keeps the image the user uploaded (for the analysis session)
together with the AI output.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    filename: Mapped[str] = mapped_column(String(255))
    image_path: Mapped[str] = mapped_column(String(512))
    grad_cam_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    prediction_label: Mapped[str] = mapped_column(String(64), index=True)
    class_index: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)
    low_confidence: Mapped[bool] = mapped_column(Boolean, default=False)
    probabilities_json: Mapped[str] = mapped_column(Text)

    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(64))

    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_checksum: Mapped[str] = mapped_column(String(64), index=True)

    report: Mapped["ReportRecord | None"] = relationship(
        back_populates="analysis", uselist=False
    )

    def to_dict(self) -> dict:
        import json

        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() + "Z",
            "filename": self.filename,
            "prediction": self.prediction_label,
            "confidence": round(self.confidence, 4),
            "probabilities": json.loads(self.probabilities_json),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "low_confidence": self.low_confidence,
            "processing_time_ms": self.processing_time_ms,
            "report_available": self.report is not None,
        }


class ModelRecord(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(64))
    architecture: Mapped[str] = mapped_column(String(128))
    file_path: Mapped[str] = mapped_column(String(512))
    input_size: Mapped[int] = mapped_column(Integer)
    dataset_version: Mapped[str] = mapped_column(String(255))
    training_datetime: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(32), default="ready")
    description: Mapped[str] = mapped_column(Text, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    class Config:
        unique_together = (("name", "version"),)

    def metric(self, key: str, default=None):
        import json

        metrics = json.loads(self.metrics_json or "{}")
        return metrics.get(key, default)

    def to_dict(self) -> dict:
        import json

        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "architecture": self.architecture,
            "file_path": self.file_path,
            "input_size": self.input_size,
            "dataset_version": self.dataset_version,
            "training_datetime": (
                self.training_datetime.isoformat() + "Z" if self.training_datetime else None
            ),
            "metrics": json.loads(self.metrics_json or "{}"),
            "status": self.status,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() + "Z",
        }


class ReportRecord(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    file_path: Mapped[str] = mapped_column(String(512))
    size_bytes: Mapped[int] = mapped_column(Integer)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    analysis: Mapped[AnalysisRecord] = relationship(back_populates="report")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "size_bytes": self.size_bytes,
            "generated_at": self.generated_at.isoformat() + "Z",
        }