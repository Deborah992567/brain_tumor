"""Inference engine.

Coordinates preprocessing, model prediction and explainability for a single
validated MRI upload. The active model is resolved centrally through the
model manager and its backend instance is reused across requests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy.orm import Session

from app.core.constants import CONFIDENCE_LOW_THRESHOLD, CLASSES, display_label
from app.services.errors import ModelUnavailableError
from app.services.model_manager import ModelManager
from app.services.preprocessing import PreparedImage, prepare_image


@dataclass
class InferenceResult:
    prediction: str
    label: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]
    model: dict
    low_confidence: bool
    processing_time_ms: int
    prepared: PreparedImage = field(repr=False)
    heatmap: np.ndarray = field(repr=False)


class InferenceService:
    def __init__(self, manager: ModelManager | None = None) -> None:
        self.manager = manager or ModelManager()
        self._model_info: dict | None = None
        self._input_size: int = 64

    def _active_model_info(self, model_record) -> dict:
        return {
            "name": model_record.name,
            "version": model_record.version,
            "architecture": model_record.architecture,
            "input_size": model_record.input_size,
            "dataset_version": model_record.dataset_version,
        }

    def analyze(self, image_bytes: bytes, db: Session) -> InferenceResult:
        model_record = self.manager.get_active(db)

        backend = self.manager.get_backend(model_record)
        input_size = model_record.input_size
        classes = list(getattr(backend, "classes", CLASSES))

        started = time.perf_counter()
        prepared = prepare_image(image_bytes, input_size=input_size)
        probabilities = backend.predict(prepared.tensor)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if len(probabilities) != len(classes):
            raise ModelUnavailableError(
                "Model output size does not match the expected number of classes."
            )

        prob_map = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        class_index = int(np.argmax(probabilities))
        prediction = classes[class_index]
        confidence = float(probabilities[class_index])
        low_confidence = confidence < CONFIDENCE_LOW_THRESHOLD

        heatmap = backend.grad_cam(prepared.tensor, class_index)

        return InferenceResult(
            prediction=prediction,
            label=display_label(prediction),
            class_index=class_index,
            confidence=confidence,
            probabilities=prob_map,
            model=self._active_model_info(model_record),
            low_confidence=low_confidence,
            processing_time_ms=elapsed_ms,
            prepared=prepared,
            heatmap=heatmap,
        )