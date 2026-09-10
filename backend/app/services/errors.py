"""Domain-level exceptions with HTTP mapping."""

from __future__ import annotations


class AppError(Exception):
    status_code = 400
    message = "An application error occurred."


class ValidationError(AppError):
    status_code = 422
    message = "Unable to analyze this image. Please upload a valid brain MRI image."

    def __init__(self, message: str | None = None, details: list[str] | None = None) -> None:
        super().__init__(message or self.message)
        self.detail = message or self.message
        self.details = details or []


class ModelUnavailableError(AppError):
    status_code = 503
    message = "The AI model is unavailable. Please try again later."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        self.detail = message or self.message


class NotFoundError(AppError):
    status_code = 404
    message = "The requested resource was not found."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        self.detail = message or self.message


class RateLimitedError(AppError):
    status_code = 429
    message = "Too many requests. Please wait and try again."


class ReportGenerationError(AppError):
    status_code = 500
    message = "The report could not be generated."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        self.detail = message or self.message