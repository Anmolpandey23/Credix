# AIBrain

Deterministic financial extraction and decision-ready risk intelligence for credit workflows.

## What It Does

- Ingests PDF, CSV, and JSON sources
- Extracts core financial fields from noisy text
- Applies range and approximation rules deterministically
- Computes ratios and structured risk flags
- Adds qualitative risk detection from narrative signals
- Produces an auditable JSON payload for scoring/decision engines

## Extraction and Confidence Rules

- Exact value -> confidence high
- Range only -> midpoint value, confidence medium
- Approximate only -> value as written, confidence low
- Range + approximate -> midpoint value, confidence low

Examples:

- `~85 - 95` -> `90`
- `around 60 to 65` -> `62.5`
- `approx 102` -> `102`

## Output Contract

Success response shape:

```json
{
  "company": "ABC MANUFACTURING PVT LTD",
  "financials": {
    "revenue": 102.0,
    "net_profit": 20.0,
    "total_debt": 95.0,
    "bank_deposits": 70.0,
    "gst_revenue": 100.0,
    "receivables": 75.0,
    "operating_cash_flow": 3.2
  },
  "ratios": {
    "cash_flow_to_profit": 0.16,
    "debt_to_revenue": 0.93,
    "receivables_to_revenue": 0.74,
    "bank_to_gst_ratio": 0.7
  },
  "risk_flags": [
    {
      "type": "revenue_mismatch",
      "severity": "high",
      "score_impact": -25,
      "metric": "bank_to_gst_ratio",
      "value": 0.7,
      "threshold": "< 0.75"
    }
  ],
  "risk_contributions": {
    "revenue_mismatch": -25
  },
  "key_risk_drivers": [
    "Revenue realization gap"
  ],
  "insights": [
    "..."
  ],
  "field_confidence": {
    "revenue": "low"
  },
  "data_quality_score": 0.71,
  "confidence": 0.71,
  "confidence_level": "medium",
  "base_score": 100,
  "final_score": 0,
  "warnings": [
    "Data reliability is moderate: X field(s) are approximate/range-derived and Y field(s) are missing; conclusions are directionally useful and should be validated against finalized financial statements."
  ]
}
```

Notes:

- `confidence` is calibrated to `data_quality_score`
- `final_score` = `base_score + sum(risk_contributions)`, capped to `0..100`

## Qualitative Risk Detection

Narrative keyword signals can add/reinforce risks:

- `related party` -> `governance_risk`
- `cash flow stressed` -> `low_cash_flow_vs_profit`
- `receivables delay` -> `high_receivables`
- `bank mismatch` -> `revenue_mismatch`

## Run

```bash
python ai-brain/main.py --company "Sample Company" --files "/path/to/file.pdf"
```

## Test

```bash
python -m pytest tests/test_ai_brain.py -q
```

Latest validated status:

- 30 tests passing

## Key Paths

- `ai-brain/main.py` - orchestration, scoring, consistency/correction layer
- `ai-brain/rag/vectorless.py` - deterministic field extraction
- `ai-brain/utils/text_utils.py` - value parsing and normalization helpers
- `ai-brain/insights/__init__.py` - ratio-driven insight generation
- `tests/test_ai_brain.py` - regression and consistency tests