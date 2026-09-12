"""Rate limiter unit tests."""

from __future__ import annotations

from app.core.security import ClientIPRateLimiter


def test_rate_limiter_allows_burst_then_blocks():
    limiter = ClientIPRateLimiter(limit_per_minute=2)
    assert limiter.allow("10.0.0.1") is True
    assert limiter.allow("10.0.0.1") is True
    assert limiter.allow("10.0.0.1") is False
    # A different IP is unaffected.
    assert limiter.allow("10.0.0.2") is True


def test_rate_limiter_reset():
    limiter = ClientIPRateLimiter(limit_per_minute=1)
    assert limiter.allow("10.0.0.1") is True
    assert limiter.allow("10.0.0.1") is False
    limiter.reset("10.0.0.1")
    assert limiter.allow("10.0.0.1") is True