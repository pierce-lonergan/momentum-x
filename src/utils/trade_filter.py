"""
MOMENTUM-X Trade Condition Filter & WebSocket Utilities

### ARCHITECTURAL CONTEXT
Node ID: utils.trade_filter
Graph Link: docs/memory/graph_state.json → "utils.trade_filter"

### RESEARCH BASIS
SIP feed includes all trades including non-market-price events.
CONSTRAINT-002 (DATA-001-EXT): Unfiltered condition codes corrupt VWAP/RVOL.
CONSTRAINT-005: WebSocket subscription frame limited to 16,385 bytes.

### CRITICAL INVARIANTS
1. Regular session: Exclude Z (out of sequence), U (extended hours), 4 (derivatively priced), C (cash sale).
2. Pre-market: Allow U/T (extended hours) but still exclude 4, Z, C, I, W.
3. WebSocket chunks: max 400 symbols per subscribe frame (well under 16KB).

Ref: ADR-004 §3 (Trade Condition Code Filter)
Ref: DATA-001-EXT CONSTRAINT-002, CONSTRAINT-005
"""

from __future__ import annotations


# ── Condition Codes (Source: SIP CTA/UTP, Alpaca API Docs) ───────────
# Ref: DATA-001-EXT CONSTRAINT-002

REGULAR_SESSION_EXCLUDED: frozenset[str] = frozenset({
    "Z",  # Sold Out of Sequence (late report, not current price)
    "U",  # Extended Hours Trade (pre/post market)
    "T",  # Extended Hours Trade (Form T)
    "4",  # Derivatively Priced (not a market price)
    "C",  # Cash Sale (special settlement)
    "I",  # Odd Lot Trade (may not reflect market)
    "W",  # Average Price Trade
})

PREMARKET_EXCLUDED: frozenset[str] = frozenset({
    "Z",  # Out of sequence — always invalid
    "4",  # Derivatively priced — always invalid
    "C",  # Cash sale — always invalid
    "I",  # Odd lot — unreliable price discovery
    "W",  # Average price — not real-time
})

# Extended hours codes that ARE valid for pre-market gap detection
EXTENDED_HOURS_VALID: frozenset[str] = frozenset({"U", "T"})


def is_valid_regular_session_trade(conditions: list[str]) -> bool:
    """
    Check if a trade's condition codes indicate a valid regular-session trade.

    A trade is valid if NONE of its conditions are in the exclusion set.
    Empty conditions or '@' (regular way) are always valid.

    Args:
        conditions: List of condition codes from Alpaca trade message 'c' field.

    Returns:
        True if trade should be included in regular-session indicators (VWAP, RVOL).

    Ref: ADR-004 §3, CONSTRAINT-002
    """
    if not conditions:
        return True

    for code in conditions:
        if code in REGULAR_SESSION_EXCLUDED:
            return False

    return True


def is_valid_premarket_trade(conditions: list[str]) -> bool:
    """
    Check if a trade is valid for pre-market gap detection.

    Pre-market analysis INCLUDES extended hours trades (U, T) since
    those ARE the pre-market activity we're scanning for. But still
    excludes out-of-sequence, derivatively priced, and cash sales.

    Args:
        conditions: List of condition codes from Alpaca trade message.

    Returns:
        True if trade should be included in pre-market gap/RVOL analysis.

    Ref: ADR-004 §3, CONSTRAINT-002
    """
    if not conditions:
        return True

    for code in conditions:
        if code in PREMARKET_EXCLUDED:
            return False

    return True


def chunk_symbols(
    symbols: list[str],
    max_per_chunk: int = 400,
) -> list[list[str]]:
    """
    Split a symbol list into chunks safe for WebSocket subscription.

    Alpaca WebSocket has a 16,385-byte initial read limit (CONSTRAINT-005).
    Each symbol in the JSON frame takes ~6-10 bytes. At 400 symbols:
    ~4,000 bytes for symbols + JSON overhead ≈ ~5KB (well under 16KB).

    Args:
        symbols: Full list of symbols to subscribe.
        max_per_chunk: Maximum symbols per subscribe frame. Default 400.

    Returns:
        List of symbol lists, each safe for a single WebSocket frame.

    Ref: ADR-004 §4, CONSTRAINT-005
    """
    if not symbols:
        return []

    return [
        symbols[i : i + max_per_chunk]
        for i in range(0, len(symbols), max_per_chunk)
    ]
