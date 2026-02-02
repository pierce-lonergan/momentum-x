"""
MOMENTUM-X Alpaca Options Data Provider

### ARCHITECTURAL CONTEXT
Node ID: data.options_provider
Graph Link: scanner.gex → data.options_provider (concrete implementation)

### RESEARCH BASIS
Provides live options chain data for GEX computation (§19).
Maps Alpaca Markets option contracts endpoint to OptionsChainEntry.

Ref: Alpaca Markets API — /v2/options/contracts
Ref: docs/research/GEX_GAMMA_EXPOSURE.md
Ref: ADR-012

### CRITICAL INVARIANTS
1. API errors → empty chain (graceful degradation per §19.7).
2. Missing greeks → default 0.0 (conservative, reduces GEX signal strength).
3. Rate-limited to 200 req/min (shared with equity data).
4. Filters to near-term expirations only (< 45 DTE).

### DESIGN DECISIONS
- Returns OptionsChainEntry (frozen dataclass) for immutability.
- Stateless: no caching. Caller manages caching if needed.
- Only fetches contracts with OI > 0 (skip illiquid strikes).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from src.scanners.gex import OptionsChainEntry, OptionsDataProvider

logger = logging.getLogger(__name__)

# Max DTE to include (filter noise from long-dated options)
_MAX_DTE = 45


class AlpacaOptionsProvider(OptionsDataProvider):
    """
    Concrete OptionsDataProvider using Alpaca Markets options API.

    Node ID: data.options_provider
    Graph Link: docs/memory/graph_state.json → "data.options_provider"

    Maps Alpaca's /v2/options/contracts endpoint to OptionsChainEntry objects.
    Filters to near-term expirations (< 45 DTE) with non-zero OI.

    Ref: ADR-012 (GEX Tiered Architecture)
    Ref: MOMENTUM_LOGIC.md §19 (Gamma Exposure)
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://paper-api.alpaca.markets",
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url.rstrip("/")

    def get_chain(self, ticker: str, as_of: date) -> list[OptionsChainEntry]:
        """
        Fetch option chain from Alpaca for a ticker.

        Filters:
        - Only contracts with OI > 0.
        - Only expirations within _MAX_DTE of as_of.
        - Missing greeks default to 0.0.

        Args:
            ticker: Underlying symbol (e.g., "AAPL").
            as_of: Reference date for DTE filtering.

        Returns:
            List of OptionsChainEntry. Empty on error.

        Ref: ADR-012, §19.7 (graceful degradation)
        """
        try:
            raw = self._fetch_options_data(ticker, as_of)
            return self._parse_contracts(raw, as_of)
        except Exception as e:
            logger.warning(
                "Options chain fetch failed for %s: %s — returning empty chain",
                ticker, e,
            )
            return []

    def _fetch_options_data(self, ticker: str, as_of: date) -> dict[str, Any]:
        """
        Fetch raw options data from Alpaca API.

        In production, this makes an HTTP request to:
            GET {base_url}/v2/options/contracts?underlying_symbol={ticker}
                &expiration_date_lte={as_of + MAX_DTE}
                &status=active

        For testing, this method is patched with mock data.

        Returns:
            Dict with "option_contracts" key containing list of contracts.
        """
        import httpx

        max_expiry = as_of + timedelta(days=_MAX_DTE)

        url = f"{self._base_url}/v2/options/contracts"
        params = {
            "underlying_symbol": ticker,
            "expiration_date_lte": max_expiry.isoformat(),
            "status": "active",
            "limit": 1000,
        }
        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
        }

        response = httpx.get(url, params=params, headers=headers, timeout=10.0)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_contracts(
        raw: dict[str, Any],
        as_of: date,
    ) -> list[OptionsChainEntry]:
        """
        Parse Alpaca API response into OptionsChainEntry list.

        Handles:
        - Missing greeks → 0.0 defaults
        - Missing OI → 0 (skip)
        - Invalid option type → skip
        """
        contracts = raw.get("option_contracts", [])
        entries: list[OptionsChainEntry] = []

        for contract in contracts:
            try:
                oi = int(contract.get("open_interest", 0))
                if oi <= 0:
                    continue

                option_type = contract.get("type", "").lower()
                if option_type not in ("call", "put"):
                    continue

                greeks = contract.get("greeks", {}) or {}

                strike = float(contract.get("strike_price", 0))
                if strike <= 0:
                    continue

                # Parse expiration for DTE
                exp_str = contract.get("expiration_date", "")
                if exp_str:
                    exp_date = date.fromisoformat(exp_str)
                    dte = (exp_date - as_of).days
                    if dte < 0 or dte > _MAX_DTE:
                        continue
                else:
                    continue

                entries.append(OptionsChainEntry(
                    strike=strike,
                    expiration=exp_date,
                    option_type=option_type,
                    open_interest=oi,
                    gamma=float(greeks.get("gamma", 0.0)),
                    implied_volatility=float(
                        greeks.get("implied_volatility", 0.0)
                        if "implied_volatility" in greeks
                        else contract.get("implied_volatility", 0.0)
                    ),
                ))

            except (ValueError, TypeError, KeyError) as e:
                logger.debug("Skipping malformed contract: %s", e)
                continue

        logger.info(
            "Parsed %d option contracts for chain (from %d raw)",
            len(entries), len(contracts),
        )
        return entries
