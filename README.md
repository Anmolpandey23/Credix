# 🏦 Credix — AI Credit Appraisal & Risk Intelligence Platform

> **AI-powered, deterministic, auditable credit risk analysis and autonomous CAM generation for modern lending workflows.**
> Draft outputs only — not a lending decision. Always verify with original source documents.

---

## Table of Contents

1. [What is Credix?](#1-what-is-credix)
2. [The Two Scores Explained](#2-the-two-scores-explained)
3. [System Architecture](#3-system-architecture)
4. [5-Module Pipeline](#4-5-module-pipeline)
5. [Tech Stack](#5-tech-stack)
6. [Key Concepts](#6-key-concepts)
7. [Extraction & Confidence Rules](#7-extraction--confidence-rules)
8. [Financial Ratios Computed](#8-financial-ratios-computed)
9. [Risk Scoring Breakdown](#9-risk-scoring-breakdown)
10. [Output Contract (JSON)](#10-output-contract-json)
11. [Setup & Run](#11-setup--run)
12. [Testing](#12-testing)
13. [Key File Map](#13-key-file-map)
14. [Limitations & Disclaimers](#14-limitations--disclaimers)

---

## 1. What is Credix?

**Credix** is a **multi-module AI credit appraisal platform** that ingests company financial documents (PDFs, CSVs, JSON), extracts structured financial data via deterministic vectorless RAG, applies rule-based risk analysis via the **IntelliCredit Engine**, aggregates live market and fraud research, and autonomously produces an auditable **Credit Appraisal Memo (CAM)** — the kind a credit officer presents to an investment committee.

It does **not** rely on a black-box machine learning model to make an opaque lending decision. Instead, it runs **five deterministic, auditable modules**, each producing its own risk sub-score and verifiable evidence trail. Every metric is traceable back to the source document page or verified news signal.

---

## 2. The Two Scores Explained

IntelliCredit produces **two separate, independent scores** that must **not** be directly compared with each other because they use different scales and methodologies.

### 🧠 AI Brain Score (`final_score`)

| Property | Detail |
|---|---|
| **Range** | 0 – 100 |
| **Direction** | **Higher = Higher Risk** |
| **What it measures** | A composite signal derived from the pipeline's qualitative and quantitative analysis. Acts as an internal pipeline health/risk signal. |
| **How it's computed** | Starts at 100 (clean baseline), then **subtracts** points for risk flags (e.g. revenue mismatch, low cash flow, high receivables). Final value = `base_score + sum(risk_contributions)`, clamped to [0, 100]. |
| **Example** | `65/100` means moderate risk signals were detected |
| **Not** | A creditworthiness probability or a calibrated lending model coefficient |

### 📊 IntelliCredit Risk Score (`intellicredit_risk.final_score`)

| Property | Detail |
|---|---|
| **Range** | 0 – 100 |
| **Direction** | **Higher = Higher Risk** |
| **Risk Categories** | Low Risk (0–44) · Medium Risk (45–74) · High Risk (75–100) |
| **What it measures** | An aggregated, weighted composite of four independent risk sub-modules: Financial, Fraud-Indicator, External, and Operational. |
| **How it's computed** | Weighted average of four sub-scores (see Section 9) |
| **Example** | `55.65/100 — Medium Risk` means enhanced due diligence is warranted |
| **Not** | A probability of default, a credit rating, or a substitute for underwriting judgment |

> ⚠️ **The two scores are on the same 0–100 numerical scale but are produced by entirely different calculations. A value of 65 on one scale has no equivalence to 65 on the other.**

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend (app.py)                 │
│   Upload · Config · Results Tabs · CAM · Download              │
└───────────────────────────┬─────────────────────────────────────┘
                            │ AIBrainPipeline.run()
┌───────────────────────────▼─────────────────────────────────────┐
│                   ai-brain/main.py  (Orchestrator)              │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Module 1 │  │ Module 2 │  │ Module 3 │  │   Module 4    │  │
│  │Ingestion │→ │Cross-    │  │Research  │  │   Inspection  │  │
│  │+ RAG     │  │Check     │  │Agent     │  │   (Primary)   │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │               Module 5 — IntelliCredit Engine              │ │
│  │  Financial Risk · Fraud Risk · External Risk · Op. Risk   │ │
│  │  Aggregator · Explainability · CAM Generator              │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 5-Module Pipeline

### Module 1 — Multi-Source Data Ingestion Engine
- Accepts **PDF, CSV, JSON** files
- **PDF:** Extracted via `PyMuPDF` (primary), `pypdf` as fallback
- **Vectorless RAG** (`rag/vectorless.py`): Deterministic keyword-proximity extraction — no embeddings, no LLM for numbers
- **Vector RAG** (`rag/semantic.py` + `sentence-transformers`): Semantic embedding search for narrative text
- Outputs: structured `financials` dict + `field_confidence` per field

### Module 2 — Structured Cross-Checking Engine
- Compares extracted values across multiple sources/pages
- Flags discrepancies (same field, different values, same reporting period)
- Classification: `consistent | discrepancy | insufficient_evidence | incomparable`
- Missing period labels → `insufficient_evidence` (not a clean match, not a discrepancy)

### Module 3 — AI Research Agent
- **Live Web Research** via **Google News RSS** (no API key required)
- Searches: legal proceedings, regulatory fines, fraud cases, promoter records, sector news
- Articles classified by type (`legal`, `fraud`, `market`, `sector`) and severity using keyword rules
- Alternative: **Manual JSON signals** — paste a JSON array of pre-researched external signals
- Results feed into the **External Risk** sub-score

### Module 4 — Primary Inspection Integration
- Credit officers enter field observations from a physical site visit
- Inputs: capacity utilisation %, machinery condition, workforce count, findings per category
- Severity levels: `low | medium | high | critical`
- Findings feed into the **Operational Risk** sub-score

### Module 5 — Explainable AI + CAM
- **IntelliCredit Engine** aggregates four sub-scores with fixed weights
- **Explainability layer**: every score contribution → plain-English reason
- **CAM Generator** (`cam.py`): structured Credit Appraisal Memo in Markdown
- CAM sections: Executive Summary · Financials · Score Breakdown · Risk Indicators · Evidence · Cross-Check · Inspection · Research · Limitations · Reviewer Decision

---

## 5. Tech Stack

### Backend / Core Engine

| Component | Library / Approach | Why |
|---|---|---|
| **PDF text extraction** | `PyMuPDF >= 1.24.0` (primary), `pypdf >= 4.2.0` (fallback) | MuPDF is fastest and most accurate for financial PDFs; pypdf as safety net |
| **Vectorless RAG** | Custom deterministic (`rag/vectorless.py`) | Zero hallucination risk for financial numbers; rule-based keyword-proximity matching |
| **Vector / Semantic RAG** | `sentence-transformers >= 6.1.0` | All-MiniLM-L6-v2 model for narrative text semantic similarity search |
| **Cross-source reconciliation** | Custom (`cross_check/`) | Structured comparison with period/unit comparability guards |
| **Research Agent** | Google News RSS (`urllib` + `xml.etree`) | No API key; publicly available; covers legal/fraud/market news |
| **Risk scoring** | Pure Python deterministic rules | Auditable; every score traceable to a rule and source value |
| **CAM generation** | Template-based Markdown | Deterministic, reproducible; no hallucination risk |
| **Data models** | `dataclasses` / `TypedDict` | Type-safe internal contracts |
| **HTTP / SSL** | `certifi >= 2024.2.2` | Trusted CA bundle for RSS feed HTTPS |

### Frontend / UI

| Component | Library | Why |
|---|---|---|
| **Web app framework** | `Streamlit` | Rapid prototyping; built-in session state, file upload, tabs |
| **Styling** | Vanilla CSS (injected via `st.markdown`) | Full control; dark glassmorphism theme with Inter font |
| **Gauges** | Inline SVG (`_donut_svg()`) | No chart library dependency; styled donut rings |
| **Score bars** | CSS-only animated bars | Lightweight; no JS frameworks |
| **Data tables** | `pandas` + `st.dataframe` | Native Streamlit table rendering |
| **Fonts** | Google Fonts — Inter (300–900) | Premium, readable, widely used in fintech UIs |

### Testing

| Component | Library |
|---|---|
| **Unit + Integration tests** | `pytest >= 8.0.0` |
| **Coverage** | 197 tests across all modules |

---

## 6. Key Concepts

### Vectorless RAG
A Retrieval-Augmented approach that does **not** use vector embeddings or an LLM for number extraction. Instead, it uses deterministic **keyword-proximity matching**: locate field keywords (e.g. "revenue", "turnover") near numeric values in document text, apply confidence rules, and return the most credible match. This eliminates hallucination risk for financial figures.

### Vector RAG
Used for **narrative text retrieval** (e.g. finding paragraphs about related-party transactions). Uses `sentence-transformers` to embed document chunks and find semantically similar passages to a query.

### Comparability Guards
Before computing any ratio or cross-source comparison, the engine checks that values are **comparable** — same reporting period, same currency/unit, same basis. A missing period label blocks the ratio computation and raises `insufficient_evidence`. This prevents misleading scores from incomparable data.

### Data Confidence
Each extracted field carries a `confidence` level: `high | medium | low`. The overall **Data Confidence %** in the UI is the mean field confidence across all extracted fields.

- **High**: Exact numeric value found with strong keyword match
- **Medium**: Range extracted (midpoint used); or approximate keyword detected
- **Low**: Weak pattern match; value may be from a ratio/percentage table rather than absolute income statement

### External Risk via RSS Research
The Research Agent fetches recent Google News articles, classifies them by type (legal, fraud, market, sector), and scores severity. Legal disputes, regulatory fines, and data breach settlements increase the External Risk sub-score. Articles are not fed into an LLM — classification uses keyword rules.

---

## 7. Extraction & Confidence Rules

| Pattern Detected | Value Used | Confidence |
|---|---|---|
| Exact number (`33,033`) | As stated | **High** |
| Range only (`85 – 95`) | Midpoint (`90`) | **Medium** |
| Approximate only (`~102`, `approx 102`) | Value as written | **Low** |
| Range + approximate (`~85 to 95`) | Midpoint (`90`) | **Low** |
| Not found | `null` | None |

**Examples:**
```
~85 - 95         →  90,    confidence: Low
around 60 to 65  →  62.5,  confidence: Medium
approx 102       →  102,   confidence: Low
33,033           →  33033, confidence: High
```

---

## 8. Financial Ratios Computed

| Ratio | Formula | Flag Threshold |
|---|---|---|
| `cash_flow_to_profit` | Operating Cash Flow ÷ Net Profit | < 0.7 → earnings quality concern |
| `debt_to_revenue` | Total Debt ÷ Revenue | > 1.0 high; > 0.5 elevated |
| `receivables_to_revenue` | Receivables ÷ Revenue | > 0.5 high; > 0.3 elevated (slow collections) |
| `bank_to_gst_ratio` | Bank Deposits ÷ GST Revenue | < 0.75 → possible revenue-bank mismatch |
| `profit_margin` | Net Profit ÷ GST Revenue | < 5% → low profitability |
| `interest_coverage` | EBIT ÷ Interest Expense | < 1.5 → debt servicing risk |
| `dscr` | Net Operating Income ÷ Debt Payments | < 1.0 → cannot cover debt |
| `current_ratio` | Current Assets ÷ Current Liabilities | < 1.0 → liquidity risk |

> All ratios require **verified comparability** (same period, positive denominator) before being computed. Missing data produces `null`, not zero.

---

## 9. Risk Scoring Breakdown

### IntelliCredit Sub-Scores (each 0–100, higher = riskier)

| Sub-Module | File | Weight | What it scores |
|---|---|---|---|
| **Financial Risk** | `financial.py` | 40% | Current ratio, interest coverage, DSCR, profit margin, cash flow, revenue trend, debt/revenue, receivables/revenue |
| **Fraud-Risk Indicators** | `fraud.py` | 25% | GST-to-bank mismatch, profit-to-OCF divergence, revenue spike vs history |
| **External Risk** | `external.py` | 20% | Legal disputes, regulatory fines, fraud cases from Research Agent |
| **Operational Risk** | `inspection.py` + `operational.py` | 15% | Capacity utilisation, machinery, workforce, inspection findings |

### Final IntelliCredit Score Formula
```
IntelliCredit Score = (Financial × 0.40) + (Fraud × 0.25) + (External × 0.20) + (Operational × 0.15)
```

### AI Brain Score Formula
```
AI Brain Score = 100 + Σ risk_contributions   [clamped to 0–100]
```
Each risk flag (e.g. high receivables, low cash flow) contributes a negative delta. More flags = higher (riskier) AI Brain score.

---

## 10. Output Contract (JSON)

```json
{
  "company": "Infosys Limited",
  "status": "success",
  "financials": {
    "revenue": 33033.0,
    "net_profit": 7769.0,
    "total_debt": null,
    "bank_deposits": null,
    "gst_revenue": null,
    "receivables": 33781.0,
    "operating_cash_flow": 20.4
  },
  "ratios": {
    "cash_flow_to_profit": 0.0026,
    "receivables_to_revenue": 1.02,
    "debt_to_revenue": null,
    "bank_to_gst_ratio": null
  },
  "risk_flags": [
    {
      "type": "low_cash_flow_vs_profit",
      "severity": "high",
      "metric": "cash_flow_to_profit",
      "value": 0.0026,
      "threshold": "< 0.7"
    }
  ],
  "field_confidence": {
    "revenue": { "level": "low" },
    "operating_cash_flow": { "level": "high" }
  },
  "confidence": 0.57,
  "confidence_level": "low",
  "final_score": 65,
  "intellicredit_risk": {
    "status": "success",
    "final_score": 55.65,
    "risk_category": "Medium Risk",
    "sub_scores": {
      "financial": 45.0,
      "fraud": 50.0,
      "external": 10.5,
      "operational": 52.0
    },
    "reasons": ["Debt-payment burden signal unavailable…"],
    "evidence": {
      "revenue": { "value": 33033.0, "page": 3, "source": "infosys1.pdf", "evidence_status": "matched" }
    },
    "cam": { "markdown": "# Credit Appraisal Memo…", "generated_at": "2026-10-23T07:17:56Z" }
  },
  "cross_check": { "status": "insufficient_evidence" },
  "research": { "status": "success", "signals": [] },
  "warnings": ["Data reliability is low…"]
}
```

**Key fields:**
- `confidence` — mean field extraction quality (0.0–1.0)
- `confidence_level` — `low | medium | high`
- `final_score` — AI Brain score (0–100, higher = riskier)
- `intellicredit_risk.final_score` — IntelliCredit score (0–100, higher = riskier)
- `intellicredit_risk.risk_category` — `Low Risk | Medium Risk | High Risk`

---

## 11. Setup & Run

### Prerequisites
- Python ≥ 3.10
- pip

### Install

```bash
cd AIBrain
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install streamlit pandas
pip install -r requirements.txt
```

### Run Streamlit UI

```bash
source .venv/bin/activate
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
python -m streamlit run app.py --server.headless true --server.port 8501
```

Open **http://localhost:8501**

### Run CLI (headless)

```bash
python ai-brain/main.py --company "Sample Company" --files "/path/to/file.pdf"
```

---

## 12. Testing

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

Coverage: 197 tests — financial ratios, fraud signals, extraction rules, confidence scoring, CAM generation, cross-check reconciliation, research agent.

---

## 13. Key File Map

```
AIBrain/
├── app.py                                  # Streamlit frontend — all 5 modules UI
├── requirements.txt                        # Core Python dependencies
├── pyproject.toml                          # Build config, pytest, Pyright settings
│
└── ai-brain/
    ├── main.py                             # AIBrainPipeline orchestrator
    │
    ├── rag/
    │   ├── vectorless.py                   # Vectorless RAG — deterministic number extraction
    │   ├── semantic.py                     # Vector RAG — sentence-transformer embedding search
    │   └── vector.py                       # Embedding index utilities
    │
    ├── ingestion/                          # PDF/CSV/JSON file readers
    ├── extraction/                         # Post-processing, normalization
    ├── cross_check/                        # Multi-source reconciliation engine
    ├── models/                             # Shared data models / TypedDicts
    ├── utils/
    │   └── text_utils.py                   # Number parsing, normalization helpers
    ├── insights/                           # Ratio-driven insight generation
    │
    └── risk_analysis/
        ├── research_agent.py               # Orchestrates research providers
        ├── rss_research_provider.py        # Google News RSS fetcher + classifier
        ├── anthropic_research_provider.py  # (Optional) Anthropic Claude research
        ├── intellicredit_adapter.py        # Bridges pipeline output → IntelliCredit models
        │
        └── intellicredit_engine/
            ├── financial.py                # Financial Risk sub-score
            ├── fraud.py                    # Fraud-Risk Indicators sub-score
            ├── external.py                 # External Risk sub-score
            ├── inspection.py              # Operational Risk from site inspection
            ├── operational.py             # Operational Risk (missing data penalties)
            ├── aggregator.py              # Weighted aggregation of sub-scores
            ├── cam.py                     # Credit Appraisal Memo generator
            ├── explainability.py          # Score → plain-English reason mapping
            ├── validator.py               # Input validation and comparability guards
            ├── signal_merger.py           # Merges research + manual signals
            ├── models.py                  # Financials, Signals, RiskResult dataclasses
            ├── sector.py                  # Sector/industry risk overlays
            └── math_utils.py             # clamp() and safe arithmetic helpers
```

---

## 14. Limitations & Disclaimers

1. **Not a lending decision.** All outputs are draft CAMs for human review only.
2. **Extraction accuracy depends on document quality.** PDFs with Ratio Analysis tables before income statements may extract percentage values instead of absolute ₹ values.
3. **Scores are not calibrated probabilities.** Weights and thresholds are engineering defaults, not empirically validated against historical loan performance data.
4. **Research signals are not verified.** RSS articles are classified by keyword rules; source authenticity and company-identity match require independent verification.
5. **Cross-check requires period metadata.** Documents without explicit fiscal year labels produce `insufficient_evidence` — not a discrepancy, not a clean bill.
6. **Missing data ≠ fraud.** Unavailable fields raise assessment uncertainty, not automatic fraud flags.
7. **Python ≥ 3.10 required** for `match` statements and `X | Y` union type hints.

---

*IntelliCredit AI · Draft outputs only · Not a lending decision · Always verify with original documents*
