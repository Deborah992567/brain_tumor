from __future__ import annotations

from pydantic import BaseModel

from app.schemas.prediction import PredictionResponse


class HistoryItem(BaseModel):
    id: str
    created_at: str
    filename: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str
    model_version: str
    low_confidence: bool
    processing_time_ms: int | None = None
    report_available: bool


class HistoryListResponse(BaseModel):
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int
    pages: int