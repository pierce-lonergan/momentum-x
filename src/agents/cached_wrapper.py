"""
MOMENTUM-X Cached Agent Wrapper

### ARCHITECTURAL CONTEXT
Node ID: agents.cached_wrapper
Graph Link: docs/memory/graph_state.json → "agents.cached_wrapper"

### RESEARCH BASIS
LLM agents are non-deterministic by nature (temperature > 0, API variability).
For backtesting to be reproducible, agent responses must be recordable and
replayable. The CachedAgentWrapper implements the Decorator pattern:

  LIVE mode (record):
    1. Calls the underlying agent.analyze(ticker, **kwargs)
    2. Serializes the AgentSignal to JSON
    3. Stores in cache keyed by (agent_id, ticker, timestamp_bucket)

  REPLAY mode:
    1. Looks up cached response by (agent_id, ticker, timestamp_bucket)
    2. Deserializes and returns the stored AgentSignal
    3. Falls back to NEUTRAL signal if cache miss

Ref: ADR-011 (LLM-Aware Backtesting)
Ref: docs/research/CPCV_LLM_LEAKAGE.md (Section: Replay Determinism)

### CRITICAL INVARIANTS
1. Cache key uses timestamp_bucket (hour granularity) for temporal alignment.
2. REPLAY mode NEVER calls the LLM API (zero network, zero cost).
3. Cache is JSON-serializable for persistence across sessions.
4. Wrapper preserves the agent's agent_id for pipeline compatibility.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents.base import BaseAgent, AgentSignal

logger = logging.getLogger(__name__)


class CachedAgentWrapper:
    """
    Decorator that records/replays agent responses for deterministic backtesting.

    Node ID: agents.cached_wrapper
    Graph Link: docs/memory/graph_state.json → "agents.cached_wrapper"

    Modes:
        RECORD: wraps a live agent, caches every response to disk/memory
        REPLAY: returns cached responses without calling the LLM

    Cache key: (agent_id, ticker, timestamp_bucket)
    Timestamp bucket: ISO date + hour (e.g., "2025-06-15T09") for temporal alignment.

    Ref: ADR-011 (LLM-Aware Backtesting)
    """

    def __init__(
        self,
        agent: BaseAgent | None = None,
        mode: str = "record",
        cache: dict[str, dict] | None = None,
    ) -> None:
        """
        Args:
            agent: Underlying agent (required for RECORD, optional for REPLAY).
            mode: "record" (live + cache) or "replay" (cache only).
            cache: Pre-loaded cache dict. Key = cache_key, Value = serialized signal.
        """
        if mode == "record" and agent is None:
            raise ValueError("RECORD mode requires an underlying agent")

        self._agent = agent
        self._mode = mode
        self._cache: dict[str, dict] = cache or {}

    @property
    def agent_id(self) -> str:
        """Agent ID from underlying agent or cache metadata."""
        if self._agent is not None:
            return self._agent.agent_id
        # Infer from first cache entry
        for key in self._cache:
            parts = key.split("::")
            if len(parts) >= 1:
                return parts[0]
        return "unknown"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    async def analyze(self, ticker: str, **kwargs: Any) -> AgentSignal:
        """
        Analyze a ticker — either live+cache (RECORD) or from cache (REPLAY).

        Args:
            ticker: Stock symbol.
            **kwargs: Additional context (news_items, market_data, etc.).

        Returns:
            AgentSignal (live in RECORD, cached in REPLAY).
        """
        as_of = kwargs.get("as_of")
        if isinstance(as_of, datetime):
            bucket = as_of.strftime("%Y-%m-%dT%H")
        else:
            bucket = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")

        cache_key = self._make_key(self.agent_id, ticker, bucket)

        if self._mode == "replay":
            return self._replay(cache_key, ticker)
        else:
            return await self._record(cache_key, ticker, **kwargs)

    async def _record(self, cache_key: str, ticker: str, **kwargs: Any) -> AgentSignal:
        """Call live agent, cache result, return signal."""
        assert self._agent is not None

        signal = await self._agent.analyze(ticker, **kwargs)

        # Serialize to cache
        self._cache[cache_key] = self._serialize_signal(signal)

        logger.debug(
            "CACHED %s: %s → %s (conf=%.2f)",
            cache_key,
            ticker,
            signal.signal,
            signal.confidence,
        )

        return signal

    def _replay(self, cache_key: str, ticker: str) -> AgentSignal:
        """Return cached signal, or NEUTRAL fallback on cache miss."""
        cached = self._cache.get(cache_key)

        if cached is not None:
            signal = self._deserialize_signal(cached)
            logger.debug(
                "REPLAY %s: %s → %s (conf=%.2f)",
                cache_key,
                ticker,
                signal.signal,
                signal.confidence,
            )
            return signal

        # Cache miss → conservative NEUTRAL
        logger.warning("CACHE MISS %s — returning NEUTRAL fallback", cache_key)
        return AgentSignal(
            agent_id=self.agent_id,
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            signal="NEUTRAL",
            confidence=0.0,
            reasoning="Cache miss: no recorded response for this (agent, ticker, time) tuple",
            flags=["CACHE_MISS"],
            prompt_variant_id="cached",
            model_id="replay",
            latency_ms=0.0,
        )

    # ── Serialization ────────────────────────────────────────────────

    @staticmethod
    def _serialize_signal(signal: AgentSignal) -> dict:
        """Convert AgentSignal to JSON-serializable dict."""
        return {
            "agent_id": signal.agent_id,
            "ticker": signal.ticker,
            "timestamp": signal.timestamp.isoformat(),
            "signal": signal.signal,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
            "flags": signal.flags,
            "prompt_variant_id": signal.prompt_variant_id,
            "model_id": signal.model_id,
            "latency_ms": signal.latency_ms,
        }

    @staticmethod
    def _deserialize_signal(data: dict) -> AgentSignal:
        """Reconstruct AgentSignal from cached dict."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts)
        else:
            timestamp = datetime.now(timezone.utc)

        return AgentSignal(
            agent_id=data.get("agent_id", "unknown"),
            ticker=data.get("ticker", "UNKNOWN"),
            timestamp=timestamp,
            signal=data.get("signal", "NEUTRAL"),
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", ""),
            flags=data.get("flags", []),
            prompt_variant_id=data.get("prompt_variant_id", "cached"),
            model_id=data.get("model_id", "replay"),
            latency_ms=data.get("latency_ms", 0.0),
        )

    @staticmethod
    def _make_key(agent_id: str, ticker: str, bucket: str) -> str:
        """Build cache key: agent_id::ticker::timestamp_bucket."""
        return f"{agent_id}::{ticker}::{bucket}"

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save cache to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._cache, indent=2, default=str))
        logger.info("Saved %d cached responses to %s", len(self._cache), path)

    @classmethod
    def load(cls, path: str | Path, agent: BaseAgent | None = None, mode: str = "replay") -> CachedAgentWrapper:
        """Load cache from JSON file."""
        path = Path(path)
        cache = json.loads(path.read_text())
        logger.info("Loaded %d cached responses from %s", len(cache), path)
        return cls(agent=agent, mode=mode, cache=cache)

    def to_dict(self) -> dict:
        """Export full cache as dict (for embedding in larger state files)."""
        return dict(self._cache)

    @classmethod
    def from_dict(cls, data: dict, agent: BaseAgent | None = None, mode: str = "replay") -> CachedAgentWrapper:
        """Create wrapper from dict."""
        return cls(agent=agent, mode=mode, cache=data)
