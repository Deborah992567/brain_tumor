"""Model management.

Single source of truth for registered models is ``models/registry.json``.
At startup its contents are synced into the database. Inference resolves the
active model and caches the loaded backend so the model is not reloaded for
every prediction.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import ModelRecord
from app.services.errors import ModelUnavailableError, NotFoundError
from app.services.model_backend import ModelBackend

try:
    from app.models.keras_backend import KerasModelBackend  # type: ignore
except Exception:  # pragma: no cover
    KerasModelBackend = None  # type: ignore


def _datetime_parse(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


class ModelManager:
    # Backend instances are shared across ModelManager instances so the model
    # is loaded once and reused for every prediction.
    _backends: dict[str, ModelBackend] = {}
    _backends_lock = threading.Lock()

    def __init__(self) -> None:
        self._registry_path: Path = settings.registry_path

    # ---- registry file -------------------------------------------------

    def _read_registry(self) -> dict:
        if not self._registry_path.exists():
            raise ModelUnavailableError(
                f"Model registry not found at {self._registry_path}."
            )
        with open(self._registry_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data

    def _write_registry(self, data: dict) -> None:
        with self._backends_lock:
            tmp = self._registry_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            tmp.replace(self._registry_path)

    def _set_active_in_registry(self, name: str, version: str) -> None:
        data = self._read_registry()
        for model in data.get("models", []):
            model["is_active"] = (
                model.get("name") == name and model.get("version") == version
            )
        self._write_registry(data)

    # ---- db sync ------------------------------------------------------

    def sync_db_from_registry(self, db: Session) -> None:
        from app.repositories.models import ModelRepository

        repo = ModelRepository(db)
        try:
            data = self._read_registry()
        except ModelUnavailableError:
            return

        for spec in data.get("models", []):
            existing = repo.get_by_name_version(spec["name"], spec["version"])
            if existing is None:
                record = ModelRecord(
                    name=spec["name"],
                    version=spec["version"],
                    architecture=spec.get("architecture", "unknown"),
                    file_path=self._resolve_path(spec.get("file_path", "")),
                    input_size=int(spec.get("input_size", 64)),
                    dataset_version=spec.get("dataset_version", ""),
                    training_datetime=_datetime_parse(spec.get("training_datetime")),
                    metrics_json=json.dumps(spec.get("metrics", {})),
                    status=spec.get("status", "ready"),
                    description=spec.get("description", ""),
                    is_active=bool(spec.get("is_active", False)),
                )
                repo.create(record)
            else:
                existing.metrics_json = json.dumps(spec.get("metrics", {}))
                existing.status = spec.get("status", "ready")
                existing.is_active = bool(spec.get("is_active", False))
                if spec.get("metrics") != {}:
                    db.add(existing)
                db.commit()

        # Guarantee exactly one active model.
        active = repo.get_active()
        if active is None:
            all_models = repo.list()
            if all_models:
                repo.set_active(all_models[0])
                self._set_active_in_registry(all_models[0].name, all_models[0].version)

    # ---- queries ------------------------------------------------------

    def list(self, db: Session) -> list[ModelRecord]:
        from app.repositories.models import ModelRepository

        return ModelRepository(db).list()

    def get(self, db: Session, model_id: int) -> ModelRecord:
        from app.repositories.models import ModelRepository

        record = ModelRepository(db).get(model_id)
        if record is None:
            raise NotFoundError("Model not found.")
        return record

    def get_active(self, db: Session) -> ModelRecord:
        from app.repositories.models import ModelRepository

        record = ModelRepository(db).get_active()
        if record is None:
            raise NotFoundError("No active model configured.")
        return record

    def set_active(self, db: Session, model_id: int) -> ModelRecord:
        from app.repositories.models import ModelRepository

        record = ModelRepository(db).get(model_id)
        if record is None:
            raise NotFoundError("Model not found.")
        if record.status != "ready":
            raise ModelUnavailableError(
                f"Model '{record.name}' is not ready (status: {record.status})."
            )
        ModelRepository(db).set_active(record)
        self._set_active_in_registry(record.name, record.version)
        pretty = f"{record.architecture} ({record.input_size}px)"
        return record

    def score_models(self, db: Session) -> None:
        """(Re)compute smoothed comparison metrics for the comparison table."""
        from app.repositories.models import ModelRepository

        for record in ModelRepository(db).list():
            metrics = record.metric("_comparison")
            if metrics is None:
                ModelRepository(db).update_metrics(record, {"_comparison": None})

    # ---- backend resolution -------------------------------------------

    def _resolve_path(self, file_path: str) -> str:
        p = Path(file_path)
        if not p.is_absolute():
            return str(p)
        return file_path

    def backend_abs_path(self, record: ModelRecord) -> Path:
        p = Path(record.file_path)
        if p.is_absolute():
            return p
        return settings.model_dir / p

    def get_backend(self, record: ModelRecord) -> ModelBackend:
        key = f"{record.name}::{record.version}"
        with self._backends_lock:
            backend = self._backends.get(key)
            if backend is not None:
                return backend
            if KerasModelBackend is None:  # pragma: no cover
                raise ModelUnavailableError("Keras backend is not available.")
            backend = KerasModelBackend(
                file_path=str(self.backend_abs_path(record)),
                name=record.name,
                version=record.version,
                architecture=record.architecture,
                input_size=record.input_size,
            )
            backend.load()
            self._backends[key] = backend
            return backend

    def reload(self) -> None:
        with self._backends_lock:
            for backend in self._backends.values():
                try:
                    backend.close()
                except Exception:
                    pass
            self._backends.clear()

    def shutdown(self) -> None:
        self.reload()