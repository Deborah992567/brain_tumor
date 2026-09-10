"""Security-related tests: headers, admin key, CORS."""

from __future__ import annotations


def test_security_headers_present(test_client):
    response = test_client.get("/api/v1/health")
    headers = response.headers
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "SAMEORIGIN"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"


def test_cors_preflight_allowed_origin(test_client):
    response = test_client.options(
        "/api/v1/predictions",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_disallowed_origin(test_client):
    response = test_client.options(
        "/api/v1/predictions",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    allow = response.headers.get("access-control-allow-origin")
    assert allow != "http://evil.example.com"


def test_admin_key_missing_when_disabled_style():
    from fastapi import HTTPException

    from app.core.security import require_admin_key

    # With ADMIN_API_KEY set (see conftest) an absent header must be rejected.
    try:
        require_admin_key(x_admin_key=None)
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 401 or exc.status_code == 503