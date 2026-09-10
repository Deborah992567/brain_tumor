"""Admin analytics response schemas.

Only real data is reported. When the database contains no analyses, an empty
distribution is returned rather than fabricated statistics.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DistributionItem(BaseModel):
    prediction: str
    count: int


class RecentAnalysis(BaseModel):
    id: str
    created_at: str
    filename: str
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str
    model_version: str


class ModelCard(BaseModel):
    id: int
    name: str
    version: str
    architecture: str
    input_size: int
    dataset_version: str
    status: str
    is_active: bool
    metrics: dict
    file_path: str


class AdminAnalytics(BaseModel):
    total_analyses: int
    total_reports: int
    prediction_distribution: list[DistributionItem]
    recent_analyses: list[RecentAnalysis]
    model_versions: list[ModelCard]
    active_model: ModelCard | None = None