from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import ModelRecord


class ModelRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def list(self) -> list[ModelRecord]:
        return (
            self.db.query(ModelRecord)
            .order_by(ModelRecord.is_active.desc(), ModelRecord.id.desc())
            .all()
        )

    def get(self, model_id: int) -> ModelRecord | None:
        return self.db.query(ModelRecord).filter(ModelRecord.id == model_id).first()

    def get_by_name_version(self, name: str, version: str) -> ModelRecord | None:
        return (
            self.db.query(ModelRecord)
            .filter(ModelRecord.name == name, ModelRecord.version == version)
            .first()
        )

    def get_active(self) -> ModelRecord | None:
        return (
            self.db.query(ModelRecord).filter(ModelRecord.is_active.is_(True)).first()
        )

    def create(self, model: ModelRecord) -> ModelRecord:
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return model

    def set_active(self, model: ModelRecord) -> ModelRecord:
        self.db.query(ModelRecord).filter(ModelRecord.is_active.is_(True)).update(
            {ModelRecord.is_active: False}
        )
        model.is_active = True
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return model

    def update_metrics(self, model: ModelRecord, metrics: dict) -> ModelRecord:
        import json

        current = json.loads(model.metrics_json or "{}")
        current.update(metrics)
        model.metrics_json = json.dumps(current, indent=2)
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return model