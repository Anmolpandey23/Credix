
"""Conservative extraction of explicitly stated financial context."""

from __future__ import annotations

import re
from typing import Any


_PERIOD_RANGE_PATTERNS = [
    # FY 2025-26 or Financial Year FY 2025-26
    re.compile(
        r"\b(?:(?:financial|fiscal)\s+year\s+)?FY\s*"
        r"(\d{4})\s*[-/]\s*(\d{2,4})\b",
        re.IGNORECASE,
    ),
    # Financial Year 2025-26
    re.compile(
        r"\b(?:financial|fiscal)\s+year\s+"
        r"(\d{4})\s*[-/]\s*(\d{2,4})\b",
        re.IGNORECASE,
    ),
]

_SINGLE_PERIOD_PATTERNS = [
    re.compile(
        r"\b(?:financial|fiscal)\s+year\s+(\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bFY\s*(\d{4})\b", re.IGNORECASE),
]

_UNIT_TOKEN = (
    r"(crores?|cr|lakhs?|lacs?|thousands?|millions?)"
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

_UNIT_ALIASES = {
    "crore": "crore",
    "crores": "crore",
    "cr": "cr",
    "lakh": "lakh",
    "lakhs": "lakh",
    "lac": "lac",
    "lacs": "lac",
    "thousand": "thousand",
    "thousands": "thousand",
    "million": "million",
    "millions": "million",
    "rupee": "rupee",
    "rupees": "rupee",
    "inr": "inr",
}


def _unique_match(matches: list[str]) -> str | None:
    """Return a value only when exactly one distinct value is supported."""
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def _extract_period(text: str) -> str | None:
    """Extract one unambiguous financial year."""
    ranges: list[str] = []

    for pattern in _PERIOD_RANGE_PATTERNS:
        for match in pattern.finditer(text):
            start_year = int(match.group(1))
            end_token = match.group(2)

            if len(end_token) == 2:
                end_year = (
                    (start_year // 100) * 100 + int(end_token)
                )
                if end_year <= start_year:
                    end_year += 100
            else:
                end_year = int(end_token)

            # Reject invalid ranges rather than guessing.
            if end_year == start_year + 1:
                ranges.append(f"FY{start_year}-{end_year}")

    # Avoid interpreting the FY 2025 prefix in FY 2025-26
    # as a separate single-year declaration.
    if ranges:
        return _unique_match(ranges)

    years: list[str] = []
    for pattern in _SINGLE_PERIOD_PATTERNS:
        years.extend(
            f"FY{match.group(1)}"
            for match in pattern.finditer(text)
        )

    return _unique_match(years)


def _extract_currency(text: str) -> str | None:
    """Extract an explicit currency declaration."""
    matches: list[str] = []

    for match in _CURRENCY_PATTERN.finditer(text):
        token = match.group(0).upper()

        if token in {"₹", "RS", "RS."}:
            matches.append("INR")
        else:
            matches.append(token)

    return _unique_match(matches)


def _extract_unit(text: str) -> str | None:
    """Extract a declared unit or an unambiguous financial range unit.

    Explicit declarations such as 'Amounts in INR crore' take precedence
    as evidence, but conflicting units remain unknown.

    A bare suffix such as 'Revenue: 100 crore' is deliberately insufficient.
    Approximate/range expressions such as 'Rs. 98-102 Cr' and
    '9.5 to 10 Cr' are accepted because the unit is attached to an
    explicitly expressed financial amount or range.
    """
    declaration_patterns = [
        # "All amounts are stated in INR crore"
        re.compile(
            r"\b(?:all\s+)?(?:amounts?|figures?|values?)\s+"
            r"(?:are\s+)?(?:stated\s+|reported\s+|expressed\s+)?in\s+"
            r"(?:₹\s*|INR\s*|Rs\.?\s*)?"
            + _UNIT_TOKEN
            + r"\b",
            re.IGNORECASE,
        ),
        # "₹ in lakhs", "INR crore", "Rs. crores"
        re.compile(
            r"(?:₹|INR|Rs\.?)\s*(?:in\s+)?"
            + _UNIT_TOKEN
            + r"\b",
            re.IGNORECASE,
        ),
        # "Unit: crore"
        re.compile(
            r"\bunit\s*[:=]\s*"
            r"(rupees?|INR|thousands?|lakhs?|lacs?|crores?|cr|millions?)\b",
            re.IGNORECASE,
        ),
    ]

    declared_matches: list[str] = []

    for pattern in declaration_patterns:
        for match in pattern.finditer(text):
            declared_matches.append(
                _UNIT_ALIASES[match.group(1).lower()]
            )

    if declared_matches:
        return _unique_match(declared_matches)

    # Only accept units attached to a numeric range, not every number
    # followed by "crore"/"lakh".
    range_pattern = re.compile(
        r"""
        (?:
            \bRs\.?\s*       # Optional explicit rupee prefix
        )?
        -?\d[\d,]*(?:\.\d+)?
        \s*(?:to|[-–—])\s*
        -?\d[\d,]*(?:\.\d+)?
        \s*
        """ + _UNIT_TOKEN + r"\b",
        re.IGNORECASE | re.VERBOSE,
    )

    range_matches = [
        _UNIT_ALIASES[match.group(1).lower()]
        for match in range_pattern.finditer(text)
    ]

    # Also allow clearly approximate amount phrases such as:
    # "around 100 Cr", "approx. 100 Cr", "~100 Cr", "near about 10 Cr".
    approximate_pattern = re.compile(
        r"""
        (?:
            \b(?:approx(?:imately)?\.?|around|about|near(?:\s+about)?|
            roughly|approximately|touching|maybe)\s*
            |~\s*
        )
        -?\d[\d,]*(?:\.\d+)?
        \s*
        """ + _UNIT_TOKEN + r"\b",
        re.IGNORECASE | re.VERBOSE,
    )

    range_matches.extend(
        _UNIT_ALIASES[match.group(1).lower()]
        for match in approximate_pattern.finditer(text)
    )

    return _unique_match(range_matches)


def _extract_basis(evidence_text: str | None) -> str | None:
    """Extract basis only when explicitly labelled in field evidence."""
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
    """Extract explicit financial context conservatively.

    Period, currency and unit can be obtained from the supplied document
    segment and field evidence when the result is unambiguous. Accounting
    basis must be explicitly labelled in the field evidence itself.
    """
    if not isinstance(document_text, str):
        document_text = ""

    evidence = (
        evidence_text
        if isinstance(evidence_text, str)
        else ""
    )

    combined = f"{document_text}\n{evidence}"

    return {
        "period": _extract_period(combined),
        "currency": _extract_currency(combined),
        "unit": _extract_unit(combined),
        "basis": _extract_basis(evidence),
    }
