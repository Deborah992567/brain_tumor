"""Admin / research analytics endpoints (API-key protected)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends

from app.core.security import require_admin_key
from app.db.database import SessionLocal
from app.repositories.history import HistoryRepository
from app.schemas.admin import (
    AdminAnalytics,
    DistributionItem,
    ModelCard,
    RecentAnalysis,
)
from app.services.model_manager import ModelManager

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin_key)],
)


def _card(record) -> ModelCard:
    return ModelCard(
        id=record.id,
        name=record.name,
        version=record.version,
        architecture=record.architecture,
        input_size=record.input_size,
        dataset_version=record.dataset_version,
        status=record.status,
        is_active=record.is_active,
        metrics=json.loads(record.metrics_json or "{}"),
        file_path=record.file_path,
    )


@router.get(
    "/analytics",
    response_model=AdminAnalytics,
    summary="Research dashboard analytics (admin)",
)
def analytics() -> AdminAnalytics:
    db = SessionLocal()
    try:
        history = HistoryRepository(db)
        counts = history.counts()

        distribution = [
            DistributionItem(prediction=label, count=count)
            for label, count in sorted(
                history.prediction_distribution().items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ]

        recents = [
            RecentAnalysis(
                id=item.id,
                created_at=item.created_at.isoformat() + "Z",
                filename=item.filename,
                prediction=item.prediction_label,
                confidence=item.confidence,
                model_name=item.model_name,
                model_version=item.model_version,
            )
            for item in history.recent(limit=10)
        ]

        manager = ModelManager()
        records = manager.list(db)
        active = manager.get_active(db)

        return AdminAnalytics(
            total_analyses=counts["total_analyses"],
            total_reports=counts["total_reports"],
            prediction_distribution=distribution,
            recent_analyses=recents,
            model_versions=[_card(r) for r in records],
            active_model=_card(active) if active else None,
        )
    finally:
        db.close()