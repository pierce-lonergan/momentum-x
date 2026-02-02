"""
MOMENTUM-X GEX Hard Filter Gate

### ARCHITECTURAL CONTEXT
Node ID: scanner.gex_filter
Graph Link: scanner.gex → scanner.premarket (extends_filter)

### RESEARCH BASIS
Implements the hard filter tier from ADR-012 (GEX Tiered Architecture).
Rejects candidates with extreme positive GEX (dealers will suppress momentum).

Ref: docs/research/GEX_GAMMA_EXPOSURE.md (Signal A: High Positive GEX)
Ref: MOMENTUM_LOGIC.md §19.6 (Extended EMC Conjunction)
Ref: ADR-012

### CRITICAL INVARIANTS
1. GEX_norm > θ_reject → REJECT (hard filter, no debate).
2. GEX data unavailable → ACCEPT (graceful degradation, §19.7).
3. θ_reject default = 2.0 (from ADR-012 analysis).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def should_reject_gex(
    gex_normalized: float | None,
    threshold: float = 2.0,
) -> bool:
    """
    Hard filter: reject candidate if normalized GEX exceeds threshold.

    Extreme positive GEX means dealers are heavily long gamma and will
    sell into any rally, suppressing the momentum signal our system
    targets. These candidates have ~40% false positive rate (§19.1).

    Args:
        gex_normalized: GEX / (ADV × Spot). None if unavailable.
        threshold: Rejection threshold (default 2.0 per ADR-012).

    Returns:
        True if candidate should be rejected (extreme positive GEX).

    Ref: ADR-012, MOMENTUM_LOGIC.md §19.6
    """
    # Graceful degradation: missing data → don't filter
    if gex_normalized is None:
        return False

    if gex_normalized > threshold:
        logger.info(
            "GEX hard filter: REJECT (GEX_norm=%.3f > threshold=%.3f)",
            gex_normalized, threshold,
        )
        return True

    return False
