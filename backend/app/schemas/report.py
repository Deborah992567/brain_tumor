from __future__ import annotations

from pydantic import BaseModel


class ReportResponse(BaseModel):
    id: str
    analysis_id: str
    size_bytes: int
    generated_at: str
    download_url: str | None = None