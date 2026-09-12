"""Pytest fixtures.

Environment variables are configured before any ``app`` module is imported so
the settings cache picks up an isolated SQLite database and a temporary model
registry. Heavy model-loading tests can be skipped with:
    pytest -m "not integration"
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="btai_tests_"))
_DATA_ROOT = _TEST_ROOT / "data"

os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_DATA_ROOT / 'test.db'}")
os.environ.setdefault("DATA_DIR", str(_DATA_ROOT))
os.environ.setdefault("UPLOAD_DIR", str(_DATA_ROOT / "uploads"))
os.environ.setdefault("GRADCAM_DIR", str(_DATA_ROOT / "gradcam"))
os.environ.setdefault("REPORT_DIR", str(_DATA_ROOT / "reports"))
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("LOG_LEVEL", "CRITICAL")
os.environ.setdefault("PREDICTION_RATE_LIMIT", "100000")
os.environ.setdefault("REPORT_RATE_LIMIT", "100000")
os.environ.setdefault("HEALTH_RATE_LIMIT", "100000")

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REGISTRY_FIXTURE = _TEST_ROOT / "registry.json"

# Point REGISTRY_PATH at a copy of the real registry BEFORE any app module is
# imported so the (cached) settings resolve to this temp registry.
_source_registry = _BACKEND_DIR / "models" / "registry.json"
if _source_registry.exists():
    _REGISTRY_FIXTURE.write_text(_source_registry.read_text(encoding="utf-8"))
else:
    _REGISTRY_FIXTURE.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "Brain Tumor CNN",
                        "version": "1.0.0",
                        "architecture": "Custom CNN",
                        "file_path": "brain_tumor.h5",
                        "input_size": 64,
                        "dataset_version": "test",
                        "metrics": {},
                        "status": "ready",
                        "is_active": True,
                    }
                ]
            }
        )
    )
os.environ.setdefault("REGISTRY_PATH", str(_REGISTRY_FIXTURE))


@pytest.fixture(scope="session")
def sample_mri_bytes():
    sample = _BACKEND_DIR.parent / "samples" / "image_sample.jpg"
    if not sample.exists():
        pytest.skip("Sample MRI not available")
    return sample.read_bytes()


@pytest.fixture(scope="session")
def test_client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture()
def db_session():
    from app.db.database import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()