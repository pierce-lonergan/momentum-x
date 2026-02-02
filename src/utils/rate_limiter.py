"""
MOMENTUM-X Token Bucket Rate Limiter

### ARCHITECTURAL CONTEXT
Node ID: utils.rate_limiter
Graph Link: docs/memory/graph_state.json → "utils.rate_limiter"

### RESEARCH BASIS
Alpaca Trading API: 200 req/min hard limit (CONSTRAINT-004, DATA-001-EXT).
Token bucket with 10% safety buffer → 180 tokens/min operational cap.
Ref: ADR-004 §2 (Token Bucket Rate Limiter)

### CRITICAL INVARIANTS
1. Trading API cap: 180 req/min (10% buffer below 200 hard limit).
2. Market Data API cap: 9,000 req/min (10% buffer below 10,000).
3. Continuous refill (not per-minute reset) for smooth throughput.
4. async-native: non-blocking wait via asyncio.sleep.
"""

from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    """
    Async token bucket rate limiter for API call throttling.

    Node ID: utils.rate_limiter
    Ref: ADR-004 §2 (Rate Limiting Strategy)
    Ref: DATA-001-EXT CONSTRAINT-004

    Token bucket algorithm:
    - Bucket starts full at max_burst tokens
    - Each API call consumes 1 token via acquire()
    - Tokens refill continuously at tokens_per_minute / 60 per second
    - If bucket empty, acquire() awaits until a token is available
    - Bucket never exceeds max_burst (prevents long-idle burst explosion)

    Usage:
        limiter = TokenBucketRateLimiter(tokens_per_minute=180, max_burst=10)
        await limiter.acquire()  # blocks if necessary
        response = await client.get(...)
    """

    def __init__(
        self,
        tokens_per_minute: int = 180,
        max_burst: int = 10,
    ) -> None:
        """
        Args:
            tokens_per_minute: Sustained rate. 180 = Alpaca trading API
                               with 10% safety buffer (CONSTRAINT-004).
            max_burst: Maximum tokens available at once. Prevents
                       large burst after idle periods.
        """
        self._rate = tokens_per_minute / 60.0  # tokens per second
        self._max_burst = max_burst
        self._tokens = float(max_burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Acquire a single token. Blocks (async) if no tokens available.

        The blocking time is deterministic:
            wait = (1 - available_tokens) / rate
        This ensures smooth throughput without bursting past the limit.
        """
        async with self._lock:
            self._refill()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            # Calculate wait time for 1 token to become available
            deficit = 1.0 - self._tokens
            wait_seconds = deficit / self._rate

        # Release lock during sleep so other tasks aren't blocked unnecessarily
        await asyncio.sleep(wait_seconds)

        # Re-acquire and take the token
        async with self._lock:
            self._refill()
            self._tokens = max(0.0, self._tokens - 1.0)

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now

        self._tokens = min(
            self._max_burst,
            self._tokens + elapsed * self._rate,
        )


# ── Pre-configured instances per ADR-004 ─────────────────────────────

def trading_rate_limiter() -> TokenBucketRateLimiter:
    """
    Rate limiter for Alpaca Trading API.
    180 req/min (10% buffer below 200 hard limit).
    Burst of 10 for order submission flurries.
    Ref: CONSTRAINT-004
    """
    return TokenBucketRateLimiter(tokens_per_minute=180, max_burst=10)


def market_data_rate_limiter() -> TokenBucketRateLimiter:
    """
    Rate limiter for Alpaca Market Data API (Unlimited plan).
    9,000 req/min (10% buffer below 10,000 hard limit).
    Burst of 50 for batch snapshot requests.
    Ref: CONSTRAINT-004
    """
    return TokenBucketRateLimiter(tokens_per_minute=9000, max_burst=50)
