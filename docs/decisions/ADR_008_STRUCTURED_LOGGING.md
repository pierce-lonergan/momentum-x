# ADR-008: Structured Logging and Trade Lifecycle Observability

**Status**: ACCEPTED
**Date**: 2026-02-02
**Nodes**: utils.logging

---

## Context

Momentum-X processes high-velocity trade pipelines: scanner → 6 agents → scoring →
debate → risk → execution → trailing stop management. Debugging production issues
requires tracing a single trade's journey through the entire pipeline.

## Drivers

1. Standard Python logging loses trade context across async boundaries.
2. Multiple agents run in parallel — logs interleave without correlation.
3. SEC queries, WebSocket events, and order fills need unified trace IDs.
4. Post-session analysis requires structured log export (JSON lines).

## Decision

### 1. Correlation ID Architecture

Each trade evaluation gets a unique `trade_id` (UUID4 prefix + ticker):
`trade_id = "a1b2c3d4-AAPL"`. This ID propagates through:
- Orchestrator → all agent calls
- Scoring → debate engine
- Execution → position manager → trailing stop manager

### 2. Structured JSON Logging

Use Python's `logging` module with a custom JSON formatter:
```
{"ts": "2026-02-01T14:30:01Z", "level": "INFO", "trade_id": "a1b2c3d4-AAPL",
 "component": "news_agent", "msg": "Signal: STRONG_BULL (0.9)", "latency_ms": 1250}
```

### 3. Implementation: contextvars

Use `contextvars.ContextVar` for async-safe trade context propagation.
No external dependencies (structlog, loguru) — stdlib only for minimal footprint.

### 4. Log Levels

- DEBUG: Raw LLM responses, WebSocket frames
- INFO: Agent signals, MFCS scores, trade verdicts, order submissions
- WARNING: Rate limits, SEC fetch failures, WebSocket reconnections
- ERROR: Order rejections, circuit breaker triggers, fatal pipeline errors

## Consequences

**Positive**: Full trade lifecycle traceability. JSON logs ready for ELK/Datadog.
**Negative**: Slight overhead from JSON serialization (~0.1ms per log line).
**Trade-off**: stdlib-only means no automatic context injection — manual propagation.

## References

- Python contextvars: https://docs.python.org/3/library/contextvars.html
- 12-Factor App Logging: https://12factor.net/logs
