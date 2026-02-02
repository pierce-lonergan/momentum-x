"""
MOMENTUM-X Tests: Rate Limiter & Trade Condition Filter

Node ID: tests.unit.test_rate_limiter
Graph Link: tested_by → utils.rate_limiter

TDD: Written BEFORE implementation per TOR-P Phase 3.
"""

from __future__ import annotations

import asyncio
import time

import pytest


class TestTokenBucketRateLimiter:
    """Token bucket rate limiter per ADR-004 §2."""

    @pytest.mark.asyncio
    async def test_immediate_acquire_when_bucket_full(self):
        """Full bucket should grant tokens immediately."""
        from src.utils.rate_limiter import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(tokens_per_minute=180, max_burst=10)
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05  # Should be near-instant

    @pytest.mark.asyncio
    async def test_burst_up_to_max(self):
        """Should allow burst up to max_burst without waiting."""
        from src.utils.rate_limiter import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(tokens_per_minute=60, max_burst=5)
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # 5 tokens should be instant from burst

    @pytest.mark.asyncio
    async def test_blocks_when_empty(self):
        """Should block when bucket is empty until token refills."""
        from src.utils.rate_limiter import TokenBucketRateLimiter

        # 60 tokens/min = 1 token/sec. Burst of 1.
        limiter = TokenBucketRateLimiter(tokens_per_minute=60, max_burst=1)
        await limiter.acquire()  # Take the 1 burst token

        start = time.monotonic()
        await limiter.acquire()  # Should wait ~1 second
        elapsed = time.monotonic() - start
        assert elapsed >= 0.8  # At least 0.8s (allowing some tolerance)
        assert elapsed < 2.0  # But not too long

    @pytest.mark.asyncio
    async def test_refill_rate_correct(self):
        """Tokens should refill at the correct rate."""
        from src.utils.rate_limiter import TokenBucketRateLimiter

        # 120 tokens/min = 2 tokens/sec
        limiter = TokenBucketRateLimiter(tokens_per_minute=120, max_burst=2)
        await limiter.acquire()
        await limiter.acquire()  # Empty

        await asyncio.sleep(0.55)  # Wait for ~1 token to refill
        start = time.monotonic()
        await limiter.acquire()  # Should succeed quickly
        elapsed = time.monotonic() - start
        assert elapsed < 0.2

    @pytest.mark.asyncio
    async def test_does_not_exceed_max_burst(self):
        """Tokens should cap at max_burst even after long idle."""
        from src.utils.rate_limiter import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(tokens_per_minute=6000, max_burst=3)
        await asyncio.sleep(0.1)  # Let many tokens "refill"

        # Should only have 3 burst tokens max
        for _ in range(3):
            await limiter.acquire()

        # 4th should require waiting
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.005  # Some wait required


class TestTradeConditionFilter:
    """Trade condition code filtering per ADR-004 §3, CONSTRAINT-002."""

    def test_standard_trade_passes(self):
        """Standard market trade (empty conditions or '@') should pass."""
        from src.utils.trade_filter import is_valid_regular_session_trade

        assert is_valid_regular_session_trade([]) is True
        assert is_valid_regular_session_trade(["@"]) is True

    def test_out_of_sequence_excluded(self):
        """Code 'Z' (out of sequence) should be filtered."""
        from src.utils.trade_filter import is_valid_regular_session_trade

        assert is_valid_regular_session_trade(["Z"]) is False

    def test_extended_hours_excluded_from_regular(self):
        """Code 'U' (extended hours) excluded from regular session."""
        from src.utils.trade_filter import is_valid_regular_session_trade

        assert is_valid_regular_session_trade(["U"]) is False

    def test_derivatively_priced_excluded(self):
        """Code '4' (derivatively priced) should be filtered."""
        from src.utils.trade_filter import is_valid_regular_session_trade

        assert is_valid_regular_session_trade(["4"]) is False

    def test_cash_sale_excluded(self):
        """Code 'C' (cash sale) should be filtered."""
        from src.utils.trade_filter import is_valid_regular_session_trade

        assert is_valid_regular_session_trade(["C"]) is False

    def test_mixed_conditions_any_excluded_fails(self):
        """If ANY condition is excluded, trade is invalid."""
        from src.utils.trade_filter import is_valid_regular_session_trade

        assert is_valid_regular_session_trade(["@", "Z"]) is False

    def test_extended_hours_valid_for_premarket(self):
        """'U' trades ARE valid for pre-market gap detection."""
        from src.utils.trade_filter import is_valid_premarket_trade

        assert is_valid_premarket_trade(["U"]) is True
        assert is_valid_premarket_trade(["T"]) is True

    def test_derivatively_priced_invalid_for_premarket(self):
        """Derivatively priced trades invalid even in pre-market."""
        from src.utils.trade_filter import is_valid_premarket_trade

        assert is_valid_premarket_trade(["4"]) is False

    def test_chunk_symbols(self):
        """Symbol list should be chunked for WebSocket per CONSTRAINT-005."""
        from src.utils.trade_filter import chunk_symbols

        symbols = [f"SYM{i}" for i in range(1000)]
        chunks = chunk_symbols(symbols, max_per_chunk=400)
        assert len(chunks) == 3  # 400 + 400 + 200
        assert len(chunks[0]) == 400
        assert len(chunks[2]) == 200
