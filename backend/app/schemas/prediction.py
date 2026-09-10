from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    analysis_id: str
    prediction: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    low_confidence: bool
    probabilities: dict[str, float]
    model: dict
    processing_time_ms: int | None = None
    created_at: str
    report: "ReportInfo | None" = None


class ReportInfo(BaseModel):
    id: str
    available: bool


PredictionResponse.model_rebuild()