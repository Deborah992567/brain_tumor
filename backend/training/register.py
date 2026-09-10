"""Register a trained model into ``models/registry.json``.

The registry is the single source of truth for registered models. Metrics are
written verbatim from the evaluation report so the comparison table in the UI
shows real numbers only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "models" / "registry.json"


def register_model(
    name: str,
    version: str,
    architecture: str,
    file_path: str,
    input_size: int,
    dataset_version: str,
    metrics: dict,
    status: str = "ready",
    description: str = "",
    set_active: bool = False,
    registry_path: Path | None = None,
) -> None:
    path = registry_path or REGISTRY_PATH
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"models": []}

    models = data.get("models", [])
    for model in models:
        if model.get("name") == name and model.get("version") == version:
            raise ValueError(
                f"Model '{name} v{version}' is already registered. "
                "Use a new version before registering."
            )
        if set_active:
            model["is_active"] = False

    models.append(
        {
            "name": name,
            "version": version,
            "architecture": architecture,
            "file_path": file_path,
            "input_size": input_size,
            "dataset_version": dataset_version,
            "training_datetime": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "status": status,
            "description": description,
            "is_active": set_active,
        }
    )
    data["models"] = models
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    print(f"Registered '{name} v{version}' in {path} (active={set_active}).")


def list_registered(registry_path: Path | None = None) -> list[dict]:
    path = registry_path or REGISTRY_PATH
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("models", [])