# ADR-001: Multi-Agent Debate Architecture with DeepSeek-R1 Reasoning Kernel

**Status**: ACCEPTED
**Date**: 2026-02-01
**Author**: Pierce (Momentum-X)
**Protocol**: TR-P §III.2

---

## Context

We need an architecture that transforms raw market data (pre-market gaps, RVOL spikes, news catalysts, SEC filings, options flow) into high-confidence trading signals for stocks likely to move +20% in a single day. Single-prompt, single-model approaches are brittle — they hallucinate facts, fail to consider counter-arguments, and produce poorly calibrated confidence scores. The research literature (REF-001, REF-002, REF-008) demonstrates that multi-agent architectures with structured debate consistently outperform single-agent systems on trading tasks.

---

## Drivers

1. **Reliability**: TradingAgents (REF-001) achieved Sharpe 8.21 using bull/bear debate — single-agent systems plateau around Sharpe 2-3 on similar tasks.
2. **Risk Discipline**: Alpha Arena results (REF-011) show GPT-5 lost 53% from over-reliance on macro reasoning without adversarial challenge. DeepSeek's disciplined execution style preserved capital. An adversarial Risk Agent prevents unchecked bullish bias.
3. **Latency Budget**: Explosive momentum plays are time-sensitive. Pre-market scanning must complete by 9:15 AM ET. Full agent pipeline (7 agents + debate + scoring) must execute in < 90 seconds per candidate. This eliminates synchronous sequential agent chains — we need parallel dispatch with async aggregation.
4. **Cost**: Running DeepSeek-R1 671B for every agent call is cost-prohibitive at scale (~$0.55/M input tokens × 7 agents × ~4K tokens each × 20 candidates = ~$30/scan). Tiered model allocation is necessary — heavy reasoning models for synthesis, lighter models for extraction.
5. **Alpaca WebSocket Jitter**: Alpaca data streams exhibit ~1.2s latency with occasional spikes (REF: DATA-001). The scanner must buffer incoming data and operate on time-bucketed snapshots rather than individual tick events.

---

## Decision

### Architecture: Parallel Fan-Out with Debate Synthesis

```
                    ┌─────────────────────┐
     Market Data ──▶│  SCANNER ENGINE     │──▶ Candidate List
                    │  (Pure Python,      │    (max 20 stocks)
                    │   no LLM needed)    │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  ORCHESTRATOR       │
                    │  (Async Dispatch)    │
                    └──┬──┬──┬──┬──┬──┬──┘
                       │  │  │  │  │  │
            ┌──────────┘  │  │  │  │  └──────────┐
            ▼             ▼  ▼  ▼  ▼             ▼
         NEWS          TECH FUND INST         DEEP_SEARCH
         AGENT         AGENT AGENT AGENT      AGENT
      (DeepSeek       (Qwen (Qwen (Qwen    (DeepSeek
       R1-32B)        -14B) -14B) -14B)      R1-32B)
            │             │  │  │  │             │
            └──────┬──────┘  │  └──┬─────────────┘
                   │         │     │
                   ▼         ▼     ▼
            ┌─────────────────────────────┐
            │       SIGNAL AGGREGATOR     │
            │  (Weighted MFCS scoring)    │
            └─────────────┬───────────────┘
                          │
            ┌─────────────▼───────────────┐
            │       DEBATE ENGINE         │
            │  ┌─────┐  ┌─────┐  ┌─────┐ │
            │  │BULL │◄─┤JUDGE├─►│BEAR │ │
            │  │Agent│  │Agent│  │Agent│ │
            │  └─────┘  └─────┘  └─────┘ │
            │  (DeepSeek R1-32B for all)  │
            └─────────────┬───────────────┘
                          │
            ┌─────────────▼───────────────┐
            │       RISK AGENT            │
            │  (Adversarial, veto power)  │
            │  (DeepSeek R1-32B)          │
            └─────────────┬───────────────┘
                          │
            ┌─────────────▼───────────────┐
            │    EXECUTION ENGINE         │
            │    (Alpaca API)             │
            └─────────────────────────────┘
```

### Model Tiering Strategy

| Tier | Model | Role | Cost/1M tokens | Rationale |
|------|-------|------|----------------|-----------|
| **Tier 1** (Reasoning) | DeepSeek-R1-Distill-Qwen-32B | Debate agents, Risk agent, News agent | ~$0.14 (via Together AI) | Native chain-of-thought, financial reasoning strength |
| **Tier 2** (Extraction) | Qwen-2.5-14B-Instruct | Technical, Fundamental, Institutional agents | ~$0.07 | Structured data extraction, fast inference |
| **Tier 3** (Validation) | DeepSeek-R1-671B (full) | Final conviction call on top 3 candidates only | ~$0.55 | Maximum reasoning depth, used sparingly |

### Agent Communication Protocol

All agents emit a standardized `AgentSignal`:

```python
@dataclass
class AgentSignal:
    agent_id: str           # e.g., "news_agent_v1"
    ticker: str             # e.g., "NVVE"
    timestamp: datetime     # UTC
    signal: Literal["STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "STRONG_BEAR"]
    confidence: float       # [0.0, 1.0]
    reasoning: str          # Full chain-of-thought trace
    key_data: dict          # Agent-specific structured data
    flags: list[str]        # ["DILUTION_RISK", "LOW_LIQUIDITY", etc.]
    sources_used: list[str] # Data source IDs from BIBLIOGRAPHY
    prompt_variant_id: str  # For arena tracking
```

### Debate Engine Protocol

1. **Bull Agent** receives all bullish signals and constructs the strongest case for a +20% move
2. **Bear Agent** receives all signals and constructs the strongest case against
3. Both agents see each other's arguments (single round, not multi-turn — latency constraint)
4. **Judge Agent** synthesizes both cases, assigns final confidence and position sizing recommendation
5. **Risk Agent** has **veto power** — can block any trade if critical risk flags are unresolved

### Prompt Arena Integration

Every agent call is tagged with a `prompt_variant_id`. The Orchestrator can dispatch the same data to multiple prompt variants in parallel. Outcomes are tracked in the Arena scoring table. This runs in shadow mode during paper trading — all variants make predictions but only the primary variant drives execution.

---

## Consequences

### Positive
- Adversarial debate prevents unchecked bullish bias (the primary failure mode in momentum trading)
- Parallel dispatch keeps latency under 90s even with 7 agents (agents run concurrently via asyncio)
- Model tiering keeps per-scan costs to ~$2-5 (vs $30+ for full R1 on every call)
- Prompt arena enables continuous improvement without code changes
- Risk Agent veto prevents catastrophic losses from hallucinated catalysts

### Negative
- Higher system complexity vs single-agent (more failure points)
- Debate engine adds ~30s to pipeline (two additional LLM calls)
- Model tiering means extraction agents may miss nuances that Tier 1 would catch
- Arena tracking requires additional storage and evaluation infrastructure

### Mitigations
- Comprehensive health checks and circuit breakers at each pipeline stage
- Debate can be bypassed for ultra-time-sensitive halted-stock resumptions (flag: `BYPASS_DEBATE`)
- Weekly review of Tier 2 agent misses to determine if model upgrade is needed
- Arena evaluation runs async — does not block trading pipeline

---

## References

- REF-001: TradingAgents (arXiv:2412.20138) — Debate architecture, Sharpe 8.21
- REF-002: MarketSenseAI 2.0 (arXiv:2502.00415) — CoT + RAG pattern
- REF-005: DeepSeek-R1 (arXiv:2501.12948) — Reasoning kernel specification
- REF-008: TradExpert (arXiv:2411.00782) — MoE routing for specialized agents
- REF-011: Alpha Arena results — Model selection empirics
- DATA-001: Alpaca API latency characteristics
