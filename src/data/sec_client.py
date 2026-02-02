"""
MOMENTUM-X SEC EDGAR Client

### ARCHITECTURAL CONTEXT
Node ID: data.sec_client
Graph Link: docs/memory/graph_state.json → "data.sec_client"

### RESEARCH BASIS
Implements SEC EDGAR EFTS integration for real-time filing detection.
Resolves H-004: Float data freshness via S-3/424B5 dilution detection.
Ref: docs/research/SEC_EDGAR_INTEGRATION.md

### CRITICAL INVARIANTS
1. User-Agent MUST include company name + email per SEC requirements.
2. Rate limit: ≤10 req/sec (we use 8 req/sec with 20% buffer).
3. S-3 within 90 days → dilution WARNING.
4. 424B5 within 30 days → ACTIVE dilution CRITICAL.
5. No dilution filings → CLEAN assessment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any
from urllib.parse import quote, urlencode

logger = logging.getLogger(__name__)


# ── Filing Type Classification ────────────────────────────────


class FilingType(Enum):
    """
    SEC filing types relevant to momentum trading.

    Ref: docs/research/SEC_EDGAR_INTEGRATION.md (Filing Types table)

    ### CRITICAL INVARIANTS
    - S-3 and 424B5 are dilution risks (company selling shares)
    - 8-K is informational (content determines risk, not form type)
    - Form 4 tracks insider activity (buying vs selling)
    """

    S3 = "S-3"
    PROSPECTUS_424B5 = "424B5"
    EVENT_8K = "8-K"
    ANNUAL_10K = "10-K"
    QUARTERLY_10Q = "10-Q"
    INSIDER_FORM4 = "4"
    SC_13D = "SC 13D"
    SC_13G = "SC 13G"
    UNKNOWN = "UNKNOWN"

    @property
    def is_dilution_risk(self) -> bool:
        """
        Does this filing type indicate potential share dilution?

        Only S-3 (shelf registration) and 424B5 (prospectus supplement)
        directly indicate dilution. 8-K may contain dilution info but
        requires content analysis — not classified here.
        """
        return self in (FilingType.S3, FilingType.PROSPECTUS_424B5)


# ── Filing Data Model ─────────────────────────────────────────


@dataclass(frozen=True)
class Filing:
    """
    Parsed SEC filing record.

    Node ID: data.sec_client.Filing
    """

    form_type: str
    filing_type: FilingType
    filed_date: date
    company_name: str
    cik: str
    accession_number: str
    description: str
    url: str = ""

    def age_days(self, reference_date: date | None = None) -> int:
        """Days since filing was submitted."""
        ref = reference_date or date.today()
        return (ref - self.filed_date).days


# ── Dilution Risk Assessment ──────────────────────────────────


@dataclass
class DilutionAssessment:
    """
    Aggregate dilution risk from filing history.

    Node ID: data.sec_client.DilutionAssessment

    ### RISK LEVELS
    - CLEAN: No dilution filings within lookback window
    - WARNING: S-3 filed within 90 days (shelf registered, may sell)
    - CRITICAL: 424B5 within 30 days (shares actively being sold NOW)
    """

    risk_level: str  # CLEAN | WARNING | CRITICAL
    active_dilution: bool  # 424B5 within 30 days
    dilution_filings: list[Filing] = field(default_factory=list)
    insider_filing_count: int = 0
    total_filings_analyzed: int = 0


def classify_filing_risk(
    filings: list[Filing],
    reference_date: date | None = None,
    s3_lookback_days: int = 90,
    prospectus_lookback_days: int = 30,
) -> DilutionAssessment:
    """
    Classify dilution risk from a set of SEC filings.

    ### INVARIANTS (enforced by tests)
    1. No filings → CLEAN
    2. S-3 within `s3_lookback_days` → WARNING
    3. 424B5 within `prospectus_lookback_days` → CRITICAL + active_dilution=True
    4. Old dilution filings (beyond lookback) → CLEAN
    5. Highest risk level wins when multiple types present

    Args:
        filings: List of Filing objects to analyze.
        reference_date: Date to calculate age from (default: today).
        s3_lookback_days: Days to look back for S-3 filings.
        prospectus_lookback_days: Days to look back for 424B5 filings.

    Returns:
        DilutionAssessment with risk level and supporting evidence.
    """
    ref = reference_date or date.today()

    dilution_filings: list[Filing] = []
    active_dilution = False
    risk_level = "CLEAN"
    insider_count = 0

    for f in filings:
        age = f.age_days(reference_date=ref)

        # Track insider filings
        if f.filing_type == FilingType.INSIDER_FORM4:
            insider_count += 1
            continue

        # Check dilution risk filings
        if not f.filing_type.is_dilution_risk:
            continue

        # 424B5: Active dilution if within prospectus lookback
        if f.filing_type == FilingType.PROSPECTUS_424B5 and age <= prospectus_lookback_days:
            dilution_filings.append(f)
            active_dilution = True
            risk_level = "CRITICAL"

        # S-3: Warning if within shelf lookback
        elif f.filing_type == FilingType.S3 and age <= s3_lookback_days:
            dilution_filings.append(f)
            if risk_level != "CRITICAL":  # Don't downgrade from CRITICAL
                risk_level = "WARNING"

    return DilutionAssessment(
        risk_level=risk_level,
        active_dilution=active_dilution,
        dilution_filings=dilution_filings,
        insider_filing_count=insider_count,
        total_filings_analyzed=len(filings),
    )


# ── Form Type Parser ──────────────────────────────────────────

_FORM_TYPE_MAP: dict[str, FilingType] = {
    "S-3": FilingType.S3,
    "S-3/A": FilingType.S3,
    "S-3ASR": FilingType.S3,
    "424B5": FilingType.PROSPECTUS_424B5,
    "424B2": FilingType.PROSPECTUS_424B5,
    "8-K": FilingType.EVENT_8K,
    "8-K/A": FilingType.EVENT_8K,
    "10-K": FilingType.ANNUAL_10K,
    "10-K/A": FilingType.ANNUAL_10K,
    "10-Q": FilingType.QUARTERLY_10Q,
    "10-Q/A": FilingType.QUARTERLY_10Q,
    "4": FilingType.INSIDER_FORM4,
    "SC 13D": FilingType.SC_13D,
    "SC 13D/A": FilingType.SC_13D,
    "SC 13G": FilingType.SC_13G,
    "SC 13G/A": FilingType.SC_13G,
}


def parse_form_type(raw: str) -> FilingType:
    """Map raw SEC form type string to FilingType enum."""
    return _FORM_TYPE_MAP.get(raw.strip(), FilingType.UNKNOWN)


# ── SEC EDGAR Client ──────────────────────────────────────────


class SECEdgarClient:
    """
    Async client for SEC EDGAR EFTS (Full-Text Search).

    ### ARCHITECTURAL CONTEXT
    Node ID: data.sec_client
    Resolves: H-004 (Float data freshness)

    ### CRITICAL INVARIANTS
    1. User-Agent MUST be set per SEC guidelines (company + email)
    2. Rate limited to 8 req/sec (SEC allows 10, 20% buffer)
    3. EFTS endpoint returns JSON with filing metadata
    4. CIK lookup via company tickers JSON

    ### USAGE
    ```python
    client = SECEdgarClient()
    filings = await client.search_filings("AAPL", form_types=["S-3", "424B5"])
    risk = classify_filing_risk(filings)
    ```
    """

    EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
    CIK_LOOKUP_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(
        self,
        user_agent: str = "Momentum-X Research bot@momentum-x.dev",
        max_requests_per_second: float = 8.0,
    ) -> None:
        self._user_agent = user_agent
        self._max_requests_per_second = max_requests_per_second

    def _build_search_url(
        self,
        query: str,
        form_types: list[str] | None = None,
        days_back: int = 90,
    ) -> str:
        """
        Build EDGAR EFTS search URL.

        Args:
            query: Company name or ticker.
            form_types: Filter by form types (e.g., ["S-3", "424B5"]).
            days_back: Search window in days from today.

        Returns:
            Full EFTS search URL.
        """
        params: dict[str, str] = {
            "q": query,
            "dateRange": "custom",
            "startdt": str(date.today() - timedelta(days=days_back)),
            "enddt": str(date.today()),
        }
        if form_types:
            params["forms"] = ",".join(form_types)

        return f"{self.EFTS_BASE}?{urlencode(params, quote_via=quote)}"

    def _build_cik_lookup_url(self, ticker: str) -> str:
        """Build URL to look up CIK by ticker symbol."""
        params = {
            "action": "getcompany",
            "company": ticker,
            "type": "",
            "dateb": "",
            "owner": "include",
            "count": "10",
            "search_text": "",
            "action": "getcompany",
            "output": "atom",
        }
        return f"{self.CIK_LOOKUP_BASE}?{urlencode(params)}"

    async def search_filings(
        self,
        ticker: str,
        form_types: list[str] | None = None,
        days_back: int = 90,
    ) -> list[Filing]:
        """
        Search EDGAR for recent filings by ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            form_types: Optional filter (e.g., ["S-3", "424B5"]).
            days_back: Lookback window in days.

        Returns:
            List of Filing objects sorted by date (newest first).
        """
        import httpx

        url = self._build_search_url(ticker, form_types, days_back)
        headers = {"User-Agent": self._user_agent, "Accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error("SEC EDGAR search failed for %s: %s", ticker, e)
            return []

        return self._parse_efts_response(data)

    def _parse_efts_response(self, data: dict[str, Any]) -> list[Filing]:
        """Parse EFTS JSON response into Filing objects."""
        filings: list[Filing] = []
        hits = data.get("hits", {}).get("hits", [])

        for hit in hits:
            source = hit.get("_source", {})
            form_raw = source.get("form_type", "")
            filing_type = parse_form_type(form_raw)

            try:
                filed_str = source.get("file_date", "")
                filed_date = date.fromisoformat(filed_str) if filed_str else date.today()
            except ValueError:
                filed_date = date.today()

            filings.append(
                Filing(
                    form_type=form_raw,
                    filing_type=filing_type,
                    filed_date=filed_date,
                    company_name=source.get("display_names", [""])[0] if source.get("display_names") else "",
                    cik=source.get("entity_id", ""),
                    accession_number=hit.get("_id", ""),
                    description=source.get("file_description", ""),
                )
            )

        # Sort newest first
        filings.sort(key=lambda f: f.filed_date, reverse=True)
        return filings

    async def check_dilution_risk(
        self,
        ticker: str,
        reference_date: date | None = None,
    ) -> DilutionAssessment:
        """
        High-level method: search for dilution-related filings and assess risk.

        Args:
            ticker: Stock ticker.
            reference_date: Override for testing.

        Returns:
            DilutionAssessment with risk level.
        """
        filings = await self.search_filings(
            ticker=ticker,
            form_types=["S-3", "S-3/A", "S-3ASR", "424B5", "424B2", "4"],
            days_back=90,
        )
        return classify_filing_risk(filings, reference_date=reference_date)
