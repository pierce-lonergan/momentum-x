# SEC EDGAR Integration for Float Freshness and Filing Detection

**Node**: data.sec_client, agent.fundamental, agent.deep_search
**Hypothesis**: H-004 (Float data freshness)

---

## Problem

Free float data sources (FinViz, Yahoo Finance) lag by days to weeks. For explosive
momentum stocks with micro/nano floats, stale data is dangerous:

- A stock showing 5M float on FinViz may have recently filed an S-3 shelf registration
  adding 20M shares — making the float 4x larger than expected.
- An 8-K filing with a registered direct offering can dilute float overnight.

## SEC EDGAR EFTS (Full-Text Search)

EDGAR Full-Text Search System (EFTS) provides real-time access to filings:

- **Endpoint**: `https://efts.sec.gov/LATEST/search-index`
- **Rate Limit**: 10 requests/second (per User-Agent identification)
- **Filings Available**: All forms within minutes of submission

### Critical Filing Types for Momentum-X

| Form | Signal | Impact |
|------|--------|--------|
| **S-3** | Shelf registration | DILUTION WARNING: Company can sell shares at will |
| **424B5** | Prospectus supplement | ACTIVE DILUTION: Shares being sold NOW |
| **8-K** | Material events | May contain: offerings, acquisitions, bankruptcies |
| **SC 13D/G** | Institutional ownership | >5% ownership changes |
| **Form 4** | Insider trades | Insider buying = bullish; selling = context-dependent |
| **10-K/10-Q** | Annual/quarterly | Float data in "Shares Outstanding" section |

### Float Calculation from SEC Filings

```
Effective Float = Outstanding Shares - Restricted Shares - Insider Holdings - Institutional Lock-ups
```

The most reliable source is the latest 10-K/10-Q "shares outstanding" figure,
cross-referenced with recent S-3/424B5 filings for dilution events.

## Implementation Strategy

1. **Filing Search**: Query EFTS for recent filings by ticker/CIK
2. **Dilution Detection**: Flag S-3/424B5 filings within 90 days
3. **Insider Activity**: Parse Form 4 for net buy/sell signal
4. **Cross-Reference**: Compare EDGAR float data against scanner data

## Rate Limiting

SEC requires identifying User-Agent with company name and email.
10 req/sec is generous but we implement a token bucket at 8 req/sec (20% buffer).

## References

- SEC EDGAR EFTS API documentation
- SEC EDGAR Developer Resources: https://www.sec.gov/search#/dateRange=custom
- CONSTRAINT-011 (Alpaca News = Benzinga, no sentiment) — SEC provides complementary data
