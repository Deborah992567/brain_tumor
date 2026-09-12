"""Health endpoint tests."""

from __future__ import annotations


def test_health_returns_ok(test_client):
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["app_name"]
    assert body["version"]
    assert "database" in body
    assert "model" in body


def test_root_meta(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()


def test_openapi_includes_versioned_routes(test_client):
    response = test_client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/health" in paths
    assert "/api/v1/predictions" in paths
    assert "/api/v1/history" in paths
    assert "/api/v1/models" in paths
    assert "/api/v1/reports" in paths