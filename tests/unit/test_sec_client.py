"""
MOMENTUM-X Tests: SEC EDGAR Client

Node ID: tests.unit.test_sec_client
Graph Link: tested_by → data.sec_client

Tests cover:
- EFTS search URL construction
- Filing type classification (dilution vs informational)
- Dilution risk scoring
- CIK lookup
- Rate limiting compliance
- Response parsing
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.data.sec_client import (
    SECEdgarClient,
    Filing,
    FilingType,
    DilutionAssessment,
    classify_filing_risk,
)


class TestFilingType:
    """Filing type classification."""

    def test_s3_is_dilution_risk(self):
        assert FilingType.S3.is_dilution_risk is True

    def test_424b5_is_dilution_risk(self):
        assert FilingType.PROSPECTUS_424B5.is_dilution_risk is True

    def test_10k_is_not_dilution_risk(self):
        assert FilingType.ANNUAL_10K.is_dilution_risk is False

    def test_form4_is_not_dilution_risk(self):
        assert FilingType.INSIDER_FORM4.is_dilution_risk is False

    def test_8k_is_not_dilution_risk(self):
        """8-K is informational — dilution risk depends on content."""
        assert FilingType.EVENT_8K.is_dilution_risk is False

    def test_sc13d_is_not_dilution_risk(self):
        assert FilingType.SC_13D.is_dilution_risk is False


class TestFiling:
    """Filing data model."""

    def test_filing_creation(self):
        f = Filing(
            form_type="S-3",
            filing_type=FilingType.S3,
            filed_date=date(2026, 1, 15),
            company_name="Test Corp",
            cik="0001234567",
            accession_number="0001234567-26-000123",
            description="Registration Statement",
        )
        assert f.form_type == "S-3"
        assert f.filing_type.is_dilution_risk is True

    def test_filing_age_days(self):
        today = date(2026, 2, 1)
        f = Filing(
            form_type="S-3",
            filing_type=FilingType.S3,
            filed_date=date(2026, 1, 15),
            company_name="Test Corp",
            cik="0001234567",
            accession_number="0001234567-26-000123",
            description="Shelf Registration",
        )
        assert f.age_days(reference_date=today) == 17


class TestDilutionRiskClassification:
    """
    Dilution assessment from filing history.

    INV: S-3 within 90 days → dilution warning
    INV: 424B5 within 30 days → ACTIVE dilution → BEAR signal
    INV: No dilution filings → clean
    """

    def test_no_filings_is_clean(self):
        assessment = classify_filing_risk(filings=[], reference_date=date(2026, 2, 1))
        assert assessment.risk_level == "CLEAN"
        assert assessment.active_dilution is False

    def test_old_s3_is_clean(self):
        """S-3 filed > 90 days ago should not trigger warning."""
        filings = [
            Filing(
                form_type="S-3",
                filing_type=FilingType.S3,
                filed_date=date(2025, 9, 1),  # >150 days ago
                company_name="Old Corp",
                cik="000111",
                accession_number="000111-25-001",
                description="Old shelf",
            )
        ]
        assessment = classify_filing_risk(filings, reference_date=date(2026, 2, 1))
        assert assessment.risk_level == "CLEAN"

    def test_recent_s3_triggers_warning(self):
        """S-3 filed within 90 days → WARNING."""
        filings = [
            Filing(
                form_type="S-3",
                filing_type=FilingType.S3,
                filed_date=date(2026, 1, 10),  # 22 days ago
                company_name="Risky Corp",
                cik="000222",
                accession_number="000222-26-001",
                description="Shelf registration",
            )
        ]
        assessment = classify_filing_risk(filings, reference_date=date(2026, 2, 1))
        assert assessment.risk_level == "WARNING"
        assert assessment.active_dilution is False
        assert len(assessment.dilution_filings) == 1

    def test_424b5_triggers_active_dilution(self):
        """424B5 within 30 days → ACTIVE dilution → highest risk."""
        filings = [
            Filing(
                form_type="424B5",
                filing_type=FilingType.PROSPECTUS_424B5,
                filed_date=date(2026, 1, 20),  # 12 days ago
                company_name="Diluting Corp",
                cik="000333",
                accession_number="000333-26-001",
                description="Prospectus supplement",
            )
        ]
        assessment = classify_filing_risk(filings, reference_date=date(2026, 2, 1))
        assert assessment.risk_level == "CRITICAL"
        assert assessment.active_dilution is True

    def test_mixed_filings_highest_risk_wins(self):
        """When multiple filing types present, highest risk level wins."""
        filings = [
            Filing(
                form_type="10-K",
                filing_type=FilingType.ANNUAL_10K,
                filed_date=date(2026, 1, 5),
                company_name="Mixed Corp",
                cik="000444",
                accession_number="000444-26-001",
                description="Annual report",
            ),
            Filing(
                form_type="S-3",
                filing_type=FilingType.S3,
                filed_date=date(2026, 1, 20),  # Recent dilution
                company_name="Mixed Corp",
                cik="000444",
                accession_number="000444-26-002",
                description="Shelf registration",
            ),
        ]
        assessment = classify_filing_risk(filings, reference_date=date(2026, 2, 1))
        assert assessment.risk_level == "WARNING"
        assert len(assessment.dilution_filings) == 1

    def test_form4_insider_buying_detected(self):
        """Form 4 filings should be counted for insider activity."""
        filings = [
            Filing(
                form_type="4",
                filing_type=FilingType.INSIDER_FORM4,
                filed_date=date(2026, 1, 25),
                company_name="Insider Corp",
                cik="000555",
                accession_number="000555-26-001",
                description="Statement of changes in beneficial ownership",
            )
        ]
        assessment = classify_filing_risk(filings, reference_date=date(2026, 2, 1))
        assert assessment.risk_level == "CLEAN"
        assert assessment.insider_filing_count == 1


class TestSECEdgarClient:
    """Client configuration and URL construction."""

    def test_default_user_agent(self):
        client = SECEdgarClient()
        assert "momentum-x" in client._user_agent.lower() or "@" in client._user_agent

    def test_efts_search_url(self):
        client = SECEdgarClient()
        url = client._build_search_url(query="AAPL", form_types=["S-3", "424B5"], days_back=90)
        assert "efts.sec.gov" in url
        assert "AAPL" in url

    def test_cik_lookup_url(self):
        client = SECEdgarClient()
        url = client._build_cik_lookup_url(ticker="AAPL")
        assert "AAPL" in url

    def test_rate_limit_config(self):
        """SEC requires ≤10 req/sec. We use 8 req/sec (20% buffer)."""
        client = SECEdgarClient()
        assert client._max_requests_per_second <= 10
