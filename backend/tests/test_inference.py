"""Inference service tests.

A fake backend is used to verify the response *structure* without loading
TensorFlow; a separate ``integration``-marked test runs the real model once
the backend is available.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.services.inference import InferenceService

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]


class FakeRecord:
    name = "FakeNet"
    version = "9.9"
    architecture = "Fake CNN"
    input_size = 64
    dataset_version = "fake"
    file_path = "fake.keras"


class FakeBackend:
    classes = CLASSES

    def predict(self, tensor):
        probs = np.full((1, 4), 0.01, dtype=np.float32)
        probs[0, 2] = 0.97  # no_tumor dominates
        return probs

    def grad_cam(self, tensor, class_index):
        return np.full((64, 64), 0.5, dtype=np.float32)


class FakeManager:
    def __init__(self) -> None:
        self.record = FakeRecord()

    def get_active(self, db):
        return self.record

    def get_backend(self, record):
        return FakeBackend()


def test_analyze_returns_structured_result(sample_mri_bytes, db_session):
    service = InferenceService(manager=FakeManager())
    result = service.analyze(sample_mri_bytes, db_session)

    assert result.prediction == "no_tumor"
    assert result.label == "No Tumor"
    assert result.class_index == 2
    assert result.confidence == pytest.approx(0.97)
    assert set(result.probabilities.keys()) == set(CLASSES)
    assert result.low_confidence is False
    assert result.model["name"] == "FakeNet"
    assert result.model["version"] == "9.9"
    assert result.processing_time_ms >= 0
    assert result.prepared.tensor.shape == (1, 64, 64, 3)
    assert result.heatmap.shape == (64, 64)


def test_low_confidence_flag(sample_mri_bytes, db_session):
    class LowBackend(FakeBackend):
        def predict(self, tensor):
            return np.full((1, 4), 0.30, dtype=np.float32)

    class LowManager(FakeManager):
        def get_backend(self, record):
            return LowBackend()

    service = InferenceService(manager=LowManager())
    result = service.analyze(sample_mri_bytes, db_session)
    assert result.low_confidence is True


def test_probability_sums_to_one(sample_mri_bytes, db_session):
    service = InferenceService(manager=FakeManager())
    result = service.analyze(sample_mri_bytes, db_session)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.integration
def test_real_model_predicts_four_classes(sample_mri_bytes, db_session):
    """Full-stack check with the shipped CNN (requires TensorFlow)."""
    tf = pytest.importorskip("tensorflow")
    service = InferenceService()
    result = service.analyze(sample_mri_bytes, db_session)
    assert result.prediction in CLASSES
    assert result.confidence > 0.0
    assert len(result.probabilities) == 4
    assert result.heatmap.shape == (64, 64)