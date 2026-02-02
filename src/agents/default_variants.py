"""
MOMENTUM-X Default Prompt Variants

Node ID: agents.default_variants
Graph Link: agents.prompt_arena → optimizes → all agents

Seeds the Prompt Arena with 2 variants per agent (12 total).
Variant A: Structured/analytical style
Variant B: Aggressive/conviction style

After sufficient matches, the arena auto-selects winners.
Ref: docs/research/PROMPT_ARENA.md (Cold start strategy)
"""

from __future__ import annotations

from src.agents.prompt_arena import PromptArena, PromptVariant


def seed_default_variants(arena: PromptArena | None = None) -> PromptArena:
    """
    Register default prompt variants for all 6 agents.

    Each agent gets 2 variants to enable head-to-head comparison.
    Variant A: Structured, methodical analysis
    Variant B: High-conviction, concise signals

    Args:
        arena: Existing arena to populate (creates new if None).

    Returns:
        PromptArena with 12 registered variants.
    """
    if arena is None:
        arena = PromptArena()

    # ── News Agent ──
    arena.register_variant(PromptVariant(
        variant_id="news_structured_v1",
        agent_id="news_agent",
        system_prompt=(
            "You are a financial news sentiment analyst. Classify each headline's "
            "impact on short-term price movement. Focus on: catalyst strength, "
            "sector relevance, timing, and retail vs institutional narrative. "
            "Output valid JSON only."
        ),
        user_prompt_template=(
            "Analyze these headlines for {ticker} ({company_name}):\n"
            "{headlines}\n\n"
            "Market cap: {market_cap}\nSector: {sector}\n\n"
            "Provide JSON: signal, confidence, catalyst_type, key_reasoning, red_flags"
        ),
    ))
    arena.register_variant(PromptVariant(
        variant_id="news_conviction_v1",
        agent_id="news_agent",
        system_prompt=(
            "You are a momentum trader's news analyst. Your job: determine if "
            "headlines support an explosive intraday move (20%+). Be decisive. "
            "STRONG_BULL only for genuine catalysts (FDA approval, earnings beat, "
            "partnership). Ignore noise. JSON only."
        ),
        user_prompt_template=(
            "{ticker} headlines (last 24h):\n{headlines}\n\n"
            "Cap: {market_cap} | Sector: {sector}\n"
            "Is there a REAL catalyst here? JSON: signal, confidence, catalyst_type, "
            "key_reasoning, red_flags"
        ),
    ))

    # ── Technical Agent ──
    arena.register_variant(PromptVariant(
        variant_id="tech_structured_v1",
        agent_id="technical_agent",
        system_prompt=(
            "You are a technical analysis agent for intraday momentum stocks. "
            "Evaluate price action, RVOL, VWAP position, support/resistance, "
            "and volume patterns. Strict JSON output."
        ),
        user_prompt_template=(
            "Technical data for {ticker}:\n"
            "Price: ${current_price} | RVOL: {rvol}x | VWAP: ${vwap}\n"
            "Indicators: {indicators}\n\n"
            "JSON: signal, confidence, pattern, entry_zone, risk_level, key_reasoning"
        ),
    ))
    arena.register_variant(PromptVariant(
        variant_id="tech_momentum_v1",
        agent_id="technical_agent",
        system_prompt=(
            "You analyze charts for breakout trades. Key question: Is the stock "
            "setting up for continuation or exhaustion? VWAP reclaim = bullish. "
            "Volume decline = bearish. Be aggressive on setups, conservative on chop. "
            "JSON only."
        ),
        user_prompt_template=(
            "{ticker} @ ${current_price} | RVOL {rvol}x | VWAP ${vwap}\n"
            "{indicators}\n"
            "Breakout or trap? JSON: signal, confidence, pattern, entry_zone, "
            "risk_level, key_reasoning"
        ),
    ))

    # ── Fundamental Agent ──
    arena.register_variant(PromptVariant(
        variant_id="fund_structured_v1",
        agent_id="fundamental_agent",
        system_prompt=(
            "You are a fundamental analyst specializing in micro-cap float structure. "
            "Focus: float size, short interest, insider ownership, dilution risk "
            "(S-3, 424B5), and effective float calculation. Strict JSON output."
        ),
        user_prompt_template=(
            "Float structure for {ticker}:\n"
            "Float: {float_shares} | SI: {short_interest} | Insider: {insider_pct}\n"
            "SEC filings: {filings}\n\n"
            "JSON: signal, confidence, float_assessment, short_squeeze_potential, "
            "dilution_risk, key_reasoning, red_flags"
        ),
    ))
    arena.register_variant(PromptVariant(
        variant_id="fund_squeeze_v1",
        agent_id="fundamental_agent",
        system_prompt=(
            "You hunt for short squeeze setups. Key metrics: low float (<10M), "
            "high SI (>25%), no recent dilution filings, high insider ownership. "
            "Flag any S-3 or 424B5 within 90 days as immediate BEAR. JSON only."
        ),
        user_prompt_template=(
            "{ticker} squeeze analysis:\n"
            "Float: {float_shares} | Short interest: {short_interest}\n"
            "Insider: {insider_pct} | Recent filings: {filings}\n"
            "Squeeze or trap? JSON: signal, confidence, float_assessment, "
            "short_squeeze_potential, dilution_risk, key_reasoning, red_flags"
        ),
    ))

    # ── Institutional Agent ──
    arena.register_variant(PromptVariant(
        variant_id="inst_structured_v1",
        agent_id="institutional_agent",
        system_prompt=(
            "You analyze institutional flow signals: unusual options activity, "
            "dark pool prints, insider transactions. Focus on conviction indicators "
            "that precede explosive moves. JSON output only."
        ),
        user_prompt_template=(
            "Institutional flow for {ticker}:\n"
            "RVOL: {rvol}x | Options: {options_data}\n"
            "Dark pool: {dark_pool_data} | Insider trades: {insider_trades}\n\n"
            "JSON: signal, confidence, flow_direction, smart_money_indicator, "
            "key_reasoning, red_flags"
        ),
    ))
    arena.register_variant(PromptVariant(
        variant_id="inst_flow_v1",
        agent_id="institutional_agent",
        system_prompt=(
            "You detect institutional accumulation patterns. Large dark pool prints "
            "above VWAP = bullish. Unusual put activity at strikes below current = "
            "hedging (neutral). Insider buying clusters = strong conviction. JSON only."
        ),
        user_prompt_template=(
            "{ticker} flow analysis | RVOL {rvol}x\n"
            "Options: {options_data}\nDark pool: {dark_pool_data}\n"
            "Insiders: {insider_trades}\n"
            "Smart money accumulating? JSON: signal, confidence, flow_direction, "
            "smart_money_indicator, key_reasoning, red_flags"
        ),
    ))

    # ── Deep Search Agent ──
    arena.register_variant(PromptVariant(
        variant_id="deep_structured_v1",
        agent_id="deep_search_agent",
        system_prompt=(
            "You are a deep research analyst. Synthesize ALL available data: "
            "news, technicals, fundamentals, filings, social sentiment, and "
            "sector context. Provide the most comprehensive analysis. JSON output."
        ),
        user_prompt_template=(
            "Deep analysis for {ticker}:\n"
            "Candidate data: {candidate_summary}\n"
            "Agent signals so far: {prior_signals}\n"
            "SEC filings: {sec_filings}\n\n"
            "JSON: signal, confidence, thesis, contrarian_view, catalyst_timeline, "
            "key_reasoning, red_flags"
        ),
    ))
    arena.register_variant(PromptVariant(
        variant_id="deep_contrarian_v1",
        agent_id="deep_search_agent",
        system_prompt=(
            "You are a contrarian analyst. Your job: find what everyone else missed. "
            "If other agents are bullish, find the bear case. If bearish, find the "
            "bull case. Challenge consensus with evidence. JSON only."
        ),
        user_prompt_template=(
            "{ticker} contrarian deep-dive:\n"
            "Data: {candidate_summary}\nConsensus: {prior_signals}\n"
            "Filings: {sec_filings}\n"
            "What is the consensus missing? JSON: signal, confidence, thesis, "
            "contrarian_view, catalyst_timeline, key_reasoning, red_flags"
        ),
    ))

    # ── Risk Agent ──
    arena.register_variant(PromptVariant(
        variant_id="risk_structured_v1",
        agent_id="risk_agent",
        system_prompt=(
            "You are the risk management agent. Your VETO is absolute. "
            "Evaluate: position sizing risk, stop-loss adequacy, liquidity, "
            "spread, correlation, and portfolio concentration. VETO if ANY "
            "risk threshold is breached. JSON output only."
        ),
        user_prompt_template=(
            "Risk assessment for {ticker}:\n"
            "Entry: ${entry_price} | Stop: ${stop_price} | Size: {position_size}\n"
            "Spread: {spread_pct}% | Volume: {volume} | Agent signals: {signals}\n\n"
            "JSON: veto (bool), risk_level, risk_factors, position_recommendation, "
            "key_reasoning"
        ),
    ))
    arena.register_variant(PromptVariant(
        variant_id="risk_aggressive_v1",
        agent_id="risk_agent",
        system_prompt=(
            "You are an aggressive risk manager. VETO only for genuine deal-breakers: "
            "spread >3%, volume <100K, or conflicting signals. Otherwise, approve "
            "with position size guidance. Momentum stocks are volatile by nature — "
            "don't over-filter. JSON only."
        ),
        user_prompt_template=(
            "{ticker} risk check:\n"
            "Entry ${entry_price} | Stop ${stop_price} | Spread {spread_pct}%\n"
            "Vol: {volume} | Signals: {signals}\n"
            "VETO or approve? JSON: veto, risk_level, risk_factors, "
            "position_recommendation, key_reasoning"
        ),
    ))

    return arena
