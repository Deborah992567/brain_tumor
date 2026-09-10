from __future__ import annotations

from pydantic import BaseModel, Field


class ModelMeta(BaseModel):
    name: str
    version: str
    architecture: str
    input_size: int
    dataset_version: str = ""


class PredictionLinks(BaseModel):
    image: str | None = None
    gradcam: str | None = None


class PredictionResponse(BaseModel):
    analysis_id: str
    prediction: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    low_confidence: bool
    probabilities: dict[str, float]
    model: ModelMeta
    processing_time_ms: int | None = None
    created_at: str
    report_available: bool = False
    links: PredictionLinks = PredictionLinks()


class ReportInfo(BaseModel):
    id: str
    available: bool


class PredictionDetailResponse(PredictionResponse):
    filename: str | None = None
    warnings: list[str] = []
    note: str | None = None
    report: ReportInfo | None = None