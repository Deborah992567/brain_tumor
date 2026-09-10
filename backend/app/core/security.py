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
    """In-memory sliding-window rate limiter keyed by client IP."""

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