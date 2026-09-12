"""History and models API tests."""

from __future__ import annotations


def _analyze(test_client, sample_mri_bytes) -> str:
    response = test_client.post(
        "/api/v1/predictions",
        files={"file": ("mri.jpg", sample_mri_bytes, "image/jpeg")},
    )
    assert response.status_code == 201
    return response.json()["analysis_id"]


def test_history_reflects_new_analysis(test_client, sample_mri_bytes):
    _analyze(test_client, sample_mri_bytes)
    response = test_client.get("/api/v1/history")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert body["page"] == 1
    assert body["pages"] >= 1
    item = body["items"][0]
    assert item["id"]
    assert item["prediction"] in {
        "glioma",
        "meningioma",
        "no_tumor",
        "pituitary_tumor",
    }
    assert 0.0 <= item["confidence"] <= 1.0
    assert item["model_name"]
    assert "report_available" in item


def test_history_pagination_and_filter(test_client, sample_mri_bytes):
    _analyze(test_client, sample_mri_bytes)
    response = test_client.get("/api/v1/history", params={"page": 1, "page_size": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["page_size"] == 1
    assert len(body["items"]) <= 1

    filtered = test_client.get(
        "/api/v1/history", params={"prediction": "no_tumor"}
    )
    assert filtered.status_code == 200
    for item in filtered.json()["items"]:
        assert item["prediction"] == "no_tumor"


def test_history_empty_filter_returns_zero(test_client):
    response = test_client.get(
        "/api/v1/history", params={"prediction": "glioma"}
    )
    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert response.json()["items"] == []


def test_history_detail(test_client, sample_mri_bytes):
    analysis_id = _analyze(test_client, sample_mri_bytes)
    response = test_client.get(f"/api/v1/history/{analysis_id}")
    assert response.status_code == 200
    assert response.json()["id"] == analysis_id


def test_models_list_has_active_model(test_client):
    response = test_client.get("/api/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert len(body["models"]) >= 1
    assert body["active"] is not None
    assert body["active"]["is_active"] is True


def test_admin_endpoints_require_key(test_client):
    response = test_client.get("/api/v1/admin/analytics")
    assert response.status_code in {401, 503}


def test_admin_endpoint_with_wrong_key(test_client):
    response = test_client.get(
        "/api/v1/admin/analytics", headers={"X-Admin-Key": "wrong-key"}
    )
    assert response.status_code in {401, 503}


def test_admin_endpoint_with_key(test_client):
    response = test_client.get(
        "/api/v1/admin/analytics", headers={"X-Admin-Key": "test-admin-key"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "total_analyses" in body
    assert "prediction_distribution" in body
    assert "model_versions" in body
    assert "active_model" in body