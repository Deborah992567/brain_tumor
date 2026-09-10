"""Application configuration loaded from environment variables.

Credentials and environment-specific values must be provided through the
environment (see ``backend/.env.example``). No secrets are hardcoded here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Brain Tumor AI"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # Hosting
    host: str = "127.0.0.1"
    port: int = 8000

    # Data directories (resolved relative to the repository root when relative)
    data_dir: Path = REPO_DIR / "data"
    upload_dir: Path = REPO_DIR / "data" / "uploads"
    gradcam_dir: Path = REPO_DIR / "data" / "gradcam"
    report_dir: Path = REPO_DIR / "data" / "reports"
    model_dir: Path = BACKEND_DIR / "models"
    registry_path: Path = BACKEND_DIR / "models" / "registry.json"

    # Database. Defaults to SQLite so the app runs with zero external setup.
    # For persistent multi-user deployments set DATABASE_URL to MariaDB:
    #   mysql+pymysql://user:password@127.0.0.1:3306/dbname
    database_url: str = f"sqlite:///{data_dir / 'brain_tumor_ai.db'}"

    # Security
    secret_key: str = "change-me-in-production"
    admin_api_key: str = ""
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Upload constraints
    max_upload_size_mb: int = 10
    allowed_image_extensions: list[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    min_image_dimension: int = 64

    # Rate limiting (requests per minute per client)
    prediction_rate_limit: int = 6
    report_rate_limit: int = 10
    health_rate_limit: int = 30

    # Model
    active_model_name: str = "Brain Tumor CNN"
    model_timeout_seconds: int = 60

    # Reporting
    report_company_name: str = "Brain Tumor AI"
    medical_disclaimer: str = (
        "This AI-generated result is intended for research and educational "
        "assistance only and does not constitute a medical diagnosis. Always "
        "consult a qualified healthcare professional for medical interpretation."
    )
    log_level: str = "INFO"

    @field_validator(
        "data_dir",
        "upload_dir",
        "gradcam_dir",
        "report_dir",
        "model_dir",
        mode="before",
    )
    @classmethod
    def _resolve_paths(cls, value: object) -> object:
        if isinstance(value, (str, Path)):
            p = Path(value)
            if not p.is_absolute():
                return REPO_DIR / p
            return p
        return value

    @field_validator("admin_api_key")
    @classmethod
    def _strip_key(cls, value: str) -> str:
        return value.strip()

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"}

    @property
    def admin_enabled(self) -> bool:
        return bool(self.admin_api_key)

    @property
    def sqlite_database(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()