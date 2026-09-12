"""Shared API dependencies."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.security import (
    health_limiter,
    prediction_limiter,
    rate_limit_dependency,
    report_limiter,
)
from app.db.database import get_db

DbSession = Generator[Session, None, None]

health_rate_limit = Depends(
    rate_limit_dependency(health_limiter(), "Too many health checks. Please wait and try again.")
)

prediction_rate_limit = Depends(
    rate_limit_dependency(
        prediction_limiter(), "Too many analysis requests. Please wait and try again."
    )
)

report_rate_limit = Depends(
    rate_limit_dependency(report_limiter(), "Too many report requests. Please wait and try again.")
)