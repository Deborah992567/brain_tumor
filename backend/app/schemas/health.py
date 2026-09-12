from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    uptime_seconds: float
    database: str
    model: dict | None
    admin_enabled: bool