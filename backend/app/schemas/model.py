from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: int
    name: str
    version: str
    architecture: str
    file_path: str
    input_size: int
    dataset_version: str
    training_datetime: str | None = None
    metrics: dict
    status: str
    description: str = ""
    is_active: bool
    created_at: str


class ModelListResponse(BaseModel):
    models: list[ModelInfo]
    active: ModelInfo | None


class ModelWithFile(ModelInfo):
    file_content: bytes | None = None