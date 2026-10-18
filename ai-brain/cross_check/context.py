"""Conservative extraction of explicitly stated financial context."""

from __future__ import annotations

import re
from typing import Any


_PERIOD_PATTERNS = [
    # Accept both "FY 2025-26" and "Financial Year FY 2025-26".
    re.compile(
        r"\b(?:(?:financial|fiscal)\s+year\s+)?FY\s*"
        r"(\d{4})\s*[-/]\s*(\d{2,4})\b",
        re.IGNORECASE,
    ),
    # Accept "Financial Year 2025-26" and "FY 2025".
    re.compile(
        r"\b(?:(?:financial|fiscal)\s+year\s+)"
        r"(\d{4})\s*[-/]\s*(\d{2,4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:financial\s+year|fiscal\s+year|FY)\s*(\d{4})\b",
        re.IGNORECASE,
    ),
]

_UNIT_PATTERN = re.compile(
    r"\b(crores?|cr|lakhs?|lacs?|thousands?|millions?)\b",
    re.IGNORECASE,
)

_CURRENCY_PATTERN = re.compile(
    r"₹|\bINR\b|\bRs\.?(?=\s|\d)",
    re.IGNORECASE,
)

_BASIS_PATTERNS = [
    (
        re.compile(r"\bbasis\s*[:=]\s*audited\s+revenue\b", re.I),
        "audited_revenue",
    ),
    (
        re.compile(r"\bbasis\s*[:=]\s*reported\s+revenue\b", re.I),
        "reported_revenue",
    ),
    (
        re.compile(r"\bbasis\s*[:=]\s*taxable\s+turnover\b", re.I),
        "taxable_turnover",
    ),
    (
        re.compile(r"\bbasis\s*[:=]\s*gross\s+revenue\b", re.I),
        "gross_revenue",
    ),
    (
        re.compile(r"\bbasis\s*[:=]\s*net\s+revenue\b", re.I),
        "net_revenue",
    ),
]


def _unique_match(matches: list[str]) -> str | None:
    """Return a context value only when the text supports one unambiguous value."""
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def _extract_period(text: str) -> str | None:
    """Extract one unambiguous financial year without partial range matches."""
    range_patterns = [
        re.compile(
            r"\b(?:(?:financial|fiscal)\s+year\s+)?FY\s*"
            r"(\d{4})\s*[-/]\s*(\d{2,4})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:financial|fiscal)\s+year\s+"
            r"(\d{4})\s*[-/]\s*(\d{2,4})\b",
            re.IGNORECASE,
        ),
    ]

    ranges: list[str] = []
    for pattern in range_patterns:
        for match in pattern.finditer(text):
            start_year = int(match.group(1))
            end_token = match.group(2)

            if len(end_token) == 2:
                end_year = (start_year // 100) * 100 + int(end_token)
                if end_year <= start_year:
                    end_year += 100
            else:
                end_year = int(end_token)

            if end_year == start_year + 1:
                ranges.append(f"FY{start_year}-{end_year}")

    # Do not let "FY 2024" inside "FY 2024-25" count as a second period.
    if ranges:
        return _unique_match(ranges)

    single_year_patterns = [
        re.compile(
            r"\b(?:financial|fiscal)\s+year\s+(\d{4})\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bFY\s*(\d{4})\b", re.IGNORECASE),
    ]

    years: list[str] = []
    for pattern in single_year_patterns:
        years.extend(
            f"FY{match.group(1)}"
            for match in pattern.finditer(text)
        )

    return _unique_match(years)


def _extract_unit(text: str) -> str | None:
    """Extract units stated in a declaration, not merely a number's suffix."""
    declaration_patterns = [
        re.compile(
            r"\b(?:all\s+)?(?:amounts?|figures?|values?)\s+"
            r"(?:are\s+)?(?:stated\s+|reported\s+|expressed\s+)?in\s+"
            r"(?:₹\s*|INR\s*|Rs\.?\s*)?"
            r"(crores?|cr|lakhs?|lacs?|thousands?|millions?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:₹|INR|Rs\.?)\s*(?:in\s+)?"
            r"(crores?|cr|lakhs?|lacs?|thousands?|millions?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bunit\s*[:=]\s*"
            r"(rupees?|INR|thousands?|lakhs?|lacs?|crores?|cr|millions?)\b",
            re.IGNORECASE,
        ),
    ]

    matches: list[str] = []
    aliases = {
        "crore": "crore", "crores": "crore", "cr": "cr",
        "lakh": "lakh", "lakhs": "lakh", "lac": "lac", "lacs": "lac",
        "thousand": "thousand", "thousands": "thousand",
        "million": "million", "millions": "million",
        "rupee": "rupee", "rupees": "rupee", "inr": "inr",
    }

    for pattern in declaration_patterns:
        for match in pattern.finditer(text):
            matches.append(aliases[match.group(1).lower()])

    return _unique_match(matches)


def _extract_currency(text: str) -> str | None:
    matches: list[str] = []
    for match in _CURRENCY_PATTERN.finditer(text):
        token = match.group(0).upper()
        matches.append("INR" if token in {"₹", "RS", "RS."} else token)
    return _unique_match(matches)


def _extract_basis(evidence_text: str | None) -> str | None:
    """Basis must be explicitly labelled in the candidate's own evidence."""
    if not isinstance(evidence_text, str):
        return None

    matches = [
        basis
        for pattern, basis in _BASIS_PATTERNS
        if pattern.search(evidence_text)
    ]
    return _unique_match(matches)


def extract_financial_context(
    document_text: str,
    evidence_text: str | None = None,
) -> dict[str, Any]:
    """Extract explicit period/currency/unit and evidence-local accounting basis.

    Document-level period, currency and unit declarations are only used when
    unambiguous in the supplied segment. Accounting basis is not inherited
    from a document header; it must be explicitly labelled in field evidence.
    """
    if not isinstance(document_text, str):
        document_text = ""

    evidence = evidence_text if isinstance(evidence_text, str) else ""
    combined = f"{document_text}\n{evidence}"

    return {
        "period": _extract_period(combined),
        "currency": _extract_currency(combined),
        "unit": _extract_unit(combined),
        "basis": _extract_basis(evidence),
    }
