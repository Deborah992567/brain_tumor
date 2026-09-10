"""Report generation tests."""

from __future__ import annotations

import json
from datetime import datetime

import pytest


def _analyze(test_client, sample_mri_bytes) -> str:
    response = test_client.post(
        "/api/v1/predictions",
        files={"file": ("mri.jpg", sample_mri_bytes, "image/jpeg")},
    )
    assert response.status_code == 201
    return response.json()["analysis_id"]


def test_report_generation_and_download(test_client, sample_mri_bytes):
    analysis_id = _analyze(test_client, sample_mri_bytes)
    response = test_client.post("/api/v1/reports", json={"analysis_id": analysis_id})
    assert response.status_code == 201
    body = response.json()
    assert body["id"]
    assert body["analysis_id"] == analysis_id
    assert body["size_bytes"] > 1000
    assert body["download_url"]

    download = test_client.get(body["download_url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/pdf"
    assert download.content[:4] == b"%PDF"


def test_report_metadata_endpoint(test_client, sample_mri_bytes):
    analysis_id = _analyze(test_client, sample_mri_bytes)
    created = test_client.post("/api/v1/reports", json={"analysis_id": analysis_id})
    report_id = created.json()["id"]

    meta = test_client.get(f"/api/v1/reports/{report_id}")
    assert meta.status_code == 200
    assert meta.json()["id"] == report_id


def test_report_list_endpoint(test_client, sample_mri_bytes, db_session):
    analysis_id = _analyze(test_client, sample_mri_bytes)
    test_client.post("/api/v1/reports", json={"analysis_id": analysis_id})

    listing = test_client.get("/api/v1/reports")
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] >= 1
    item = next(
        (i for i in body["items"] if i["analysis_id"] == analysis_id),
        None,
    )
    assert item is not None
    assert item["prediction"] == "no_tumor"
    assert isinstance(item["confidence"], float)
    assert item["filename"] == "mri.jpg"
    assert item["download_url"].endswith("/download")


def test_report_for_missing_analysis_404(test_client):
    response = test_client.post("/api/v1/reports", json={"analysis_id": "nope"})
    assert response.status_code == 404


def test_generator_builds_pdf_with_disclaimer(tmp_path, db_session):
    """Unit-level: the report generator writes a valid PDF containing text."""
    from app.db.models import AnalysisRecord
    from app.services.reporting import ReportGenerator

    analysis = AnalysisRecord(
        id="abc-123",
        created_at=datetime(2024, 5, 1, 12, 0, 0),
        filename="mri.jpg",
        image_path="",
        prediction_label="no_tumor",
        class_index=2,
        confidence=0.91,
        low_confidence=False,
        probabilities_json=json.dumps(
            {
                "glioma": 0.01,
                "meningioma": 0.02,
                "no_tumor": 0.94,
                "pituitary_tumor": 0.03,
            }
        ),
        model_name="Test Net",
        model_version="1.0",
        image_checksum="x" * 64,
    )
    overlay = b"\x89PNG\r\n\x1a\n"  # 8-byte PNG signature; generator only wraps in reportlab
    generator = ReportGenerator(output_dir=tmp_path)
    path = generator.generate(analysis, overlay)
    assert path.exists()
    assert path.suffix == ".pdf"
    data = path.read_bytes()
    assert data[:4] == b"%PDF"
    assert b"AI-Assisted Brain MRI Analysis Report" in data