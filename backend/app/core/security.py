"""Security helpers: admin API-key dependency and security response headers."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request, status

from app.core.config import settings

ADMIN_KEY_HEADER = "X-Admin-Key"


def require_admin_key(x_admin_key: str | None = Header(default=None)) -> None:
    """Require a valid admin API key for privileged endpoints."""
    if not settings.admin_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API is disabled. Set the ADMIN_API_KEY environment variable to enable it.",
        )
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin API key.",
        )


class SecurityHeadersMiddleware:
    """Add basic security headers to every response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                security_headers = {
                    b"X-Content-Type-Options": b"nosniff",
                    b"X-Frame-Options": b"SAMEORIGIN",
                    b"Referrer-Policy": b"strict-origin-when-cross-origin",
                    b"X-XSS-Protection": b"0",
                    b"Permissions-Policy": b"camera=(), microphone=(), geolocation=()",
                }
                for name, value in security_headers.items():
                    if not any(existing.lower() == name.lower() for existing, _ in headers):
                        headers.append((name, value))
                message = dict(message)
                message["headers"] = headers
            await send(message)

        return await self.app(scope, receive, send_wrapper)


class ClientIPRateLimiter:
    """In-memory sliding-window rate limiter keyed by client IP.

    Not suitable for multi-process production deployments; replace with a
    shared store (Redis) when scaling horizontally.
    """

    def __init__(self, limit_per_minute: int, window_seconds: int = 60) -> None:
        self.limit = limit_per_minute
        self.window = window_seconds
        self._requests: dict[str, list[float]] = {}

    def allow(self, client_ip: str) -> bool:
        import time

        now = time.monotonic()
        bucket = [t for t in self._requests.get(client_ip, []) if now - t < self.window]
        if len(bucket) >= self.limit:
            self._requests[client_ip] = bucket
            return False
        bucket.append(now)
        self._requests[client_ip] = bucket
        return True

    def reset(self, client_ip: str | None = None) -> None:
        if client_ip is None:
            self._requests.clear()
        else:
            self._requests.pop(client_ip, None)


# Shared rate limiters used across routes.
_health_limiter: ClientIPRateLimiter | None = None
_prediction_limiter: ClientIPRateLimiter | None = None
_report_limiter: ClientIPRateLimiter | None = None


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit_dependency(limiter: ClientIPRateLimiter, message: str):
    """Build a FastAPI dependency that enforces a per-IP rate limit."""

    async def dependency(request: Request) -> None:
        if not limiter.allow(_client_ip(request)):
            raise HTTPException(status_code=429, detail=message)

    return dependency


def health_limiter() -> ClientIPRateLimiter:
    global _health_limiter
    if _health_limiter is None:
        _health_limiter = ClientIPRateLimiter(settings.health_rate_limit)
    return _health_limiter


def prediction_limiter() -> ClientIPRateLimiter:
    global _prediction_limiter
    if _prediction_limiter is None:
        _prediction_limiter = ClientIPRateLimiter(settings.prediction_rate_limit)
    return _prediction_limiter


def report_limiter() -> ClientIPRateLimiter:
    global _report_limiter
    if _report_limiter is None:
        _report_limiter = ClientIPRateLimiter(settings.report_rate_limit)
    return _report_limiter