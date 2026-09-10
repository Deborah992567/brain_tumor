"""End-to-end prediction API tests."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


def test_prediction_endpoint_returns_structured_response(test_client, sample_mri_bytes):
    response = test_client.post(
        "/api/v1/predictions",
        files={"file": ("mri.jpg", sample_mri_bytes, "image/jpeg")},
    )
    assert response.status_code == 201
    body = response.json()

    assert body["analysis_id"]
    assert body["prediction"] in {"glioma", "meningioma", "no_tumor", "pituitary_tumor"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == {
        "glioma",
        "meningioma",
        "no_tumor",
        "pituitary_tumor",
    }
    assert body["model"]["name"]
    assert body["model"]["version"]
    assert body["model"]["preprocessing_version"]
    assert set(body["supported_classes"]) == {
        "glioma",
        "meningioma",
        "no_tumor",
        "pituitary_tumor",
    }
    assert body["links"]["image"]
    assert body["links"]["gradcam"]


def test_prediction_media_available(test_client, sample_mri_bytes):
    analysis_id = _analyze(test_client, sample_mri_bytes)
    image = test_client.get(f"/api/v1/predictions/{analysis_id}/image")
    assert image.status_code == 200
    assert image.headers["content-type"].startswith("image/")
    grad = test_client.get(f"/api/v1/predictions/{analysis_id}/gradcam")
    assert grad.status_code == 200


def test_invalid_file_is_rejected_without_running_model(test_client):
    response = test_client.post(
        "/api/v1/predictions",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert "MRI" in body["detail"] or "format" in body["detail"]


def test_corrupt_image_rejected(test_client):
    response = test_client.post(
        "/api/v1/predictions",
        files={"file": ("bad.png", b"\x89PNG\r\n\x1a\ncorrupt", "image/png")},
    )
    assert response.status_code == 422


def test_missing_file_rejected(test_client):
    response = test_client.post("/api/v1/predictions")
    assert response.status_code == 422


def test_unknown_analysis_returns_404(test_client):
    response = test_client.get("/api/v1/predictions/does-not-exist")
    assert response.status_code == 404


def test_domain_shifted_input_returns_structured_response(test_client):
    """Regression: an unusual/out-of-distribution input (high-frequency noise,
    not resembling training MRIs) must still yield a valid, auditable,
    structured response — never a 500 and never a fabricated label."""
    rng = np.random.default_rng(11)
    pattern = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pattern).save(buf, format="PNG")

    response = test_client.post(
        "/api/v1/predictions",
        files={"file": ("synthetic_pattern.png", buf.getvalue(), "image/png")},
    )
    assert response.status_code == 201
    body = response.json()

    assert body["prediction"] in {"glioma", "meningioma", "no_tumor", "pituitary_tumor"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["low_confidence"], bool)
    total = sum(body["probabilities"].values())
    assert total == pytest.approx(1.0, abs=1e-4)
    assert set(body["supported_classes"]) == {
        "glioma",
        "meningioma",
        "no_tumor",
        "pituitary_tumor",
    }
    # Reliability-critical wording: results are model probabilities.
    assert body["model"]["preprocessing_version"]
    detail = test_client.get(f"/api/v1/predictions/{body['analysis_id']}")
    assert detail.status_code == 200
    assert "uncalibrated" in detail.json()["note"].lower()


def _analyze(test_client, sample_mri_bytes) -> str:
    response = test_client.post(
        "/api/v1/predictions",
        files={"file": ("mri.jpg", sample_mri_bytes, "image/jpeg")},
    )
    assert response.status_code == 201
    return response.json()["analysis_id"]