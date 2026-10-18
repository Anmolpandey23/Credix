"""Conservative, auditable reconciliation of financial observations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Literal


Status = Literal[
    "consistent",
    "discrepancy",
    "incomparable",
    "insufficient_evidence",
]


# Multipliers convert supported units into base currency units.
_UNIT_MULTIPLIERS = {
    "rupee": 1.0,
    "rupees": 1.0,
    "inr": 1.0,
    "rs": 1.0,
    "₹": 1.0,
    "thousand": 1_000.0,
    "thousands": 1_000.0,
    "lakh": 100_000.0,
    "lakhs": 100_000.0,
    "lac": 100_000.0,
    "crore": 10_000_000.0,
    "crores": 10_000_000.0,
    "cr": 10_000_000.0,
    "million": 1_000_000.0,
    "millions": 1_000_000.0,
}


def _normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None

    normalized = unit.strip().lower().rstrip(".")
    return normalized if normalized in _UNIT_MULTIPLIERS else None


def _normalize_currency(currency: str | None) -> str | None:
    if not currency:
        return None

    normalized = currency.strip().upper()
    aliases = {
        "₹": "INR",
        "RS": "INR",
        "RS.": "INR",
        "RUPEE": "INR",
        "RUPEES": "INR",
    }
    normalized = aliases.get(normalized, normalized)

    # Do not attempt currency conversion.
    if len(normalized) != 3 or not normalized.isalpha():
        return None

    return normalized


@dataclass(frozen=True)
class FinancialObservation:
    """One explicitly sourced financial value."""

    field: str
    value: float
    source: str
    period: str | None
    currency: str | None
    unit: str | None
    basis: str | None
    page: int | None = None
    evidence_text: str | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    """Auditable result for a pair of observations."""

    field: str
    source_a: str
    source_b: str
    page_a: int | None
    page_b: int | None
    value_a_normalized: float | None
    value_b_normalized: float | None
    difference_ratio: float | None
    status: Status
    reason: str


def reconcile_observations(
    observations: list[FinancialObservation],
    tolerance: float = 0.05,
) -> list[ReconciliationResult]:
    """Compare same-field observations only when their context is compatible.

    The difference ratio is absolute difference divided by the larger
    absolute value. A zero-versus-zero comparison has difference zero.
    Values are normalized to base currency units before comparison.

    A discrepancy is an investigation signal, not proof of misconduct.
    """
    if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
        raise ValueError("tolerance must be finite and between 0 and 1")

    # Reject malformed observations before comparing any financial values.
    for index, observation in enumerate(observations):
        if not isinstance(observation.field, str) or not observation.field.strip():
            raise ValueError(f"Observation {index}: field must not be blank")

        if not isinstance(observation.source, str) or not observation.source.strip():
            raise ValueError(f"Observation {index}: source must not be blank")

        if isinstance(observation.value, bool):
            raise ValueError(f"Observation {index}: value must be numeric, not boolean")

        if not isinstance(observation.value, (int, float)):
            raise ValueError(
                f"Observation {index}: value must be numeric (int or float)"
            )

        numeric_value = float(observation.value)

        if not math.isfinite(numeric_value):
            raise ValueError(
                f"Observation {index}: value must be finite"
            )

        if observation.page is not None:
            if (
                isinstance(observation.page, bool)
                or not isinstance(observation.page, int)
                or observation.page < 1
            ):
                raise ValueError(
                    f"Observation {index}: page must be a positive integer or None"
                )

    results: list[ReconciliationResult] = []

    for left, right in combinations(observations, 2):
        if left.field != right.field:
            continue

        result = _compare_pair(left, right, tolerance)
        results.append(result)

    return results


def _compare_pair(
    left: FinancialObservation,
    right: FinancialObservation,
    tolerance: float,
) -> ReconciliationResult:
    def result(
        status: Status,
        reason: str,
        value_a: float | None = None,
        value_b: float | None = None,
        ratio: float | None = None,
    ) -> ReconciliationResult:
        return ReconciliationResult(
            field=left.field,
            source_a=left.source,
            source_b=right.source,
            page_a=left.page,
            page_b=right.page,
            value_a_normalized=value_a,
            value_b_normalized=value_b,
            difference_ratio=ratio,
            status=status,
            reason=reason,
        )

    if (
        not isinstance(left.period, str)
        or not left.period.strip()
        or not isinstance(right.period, str)
        or not right.period.strip()
    ):
        return result(
            "insufficient_evidence",
            "Reporting period is missing from at least one observation.",
        )

    if left.period.strip() != right.period.strip():
        return result(
            "incomparable",
            "Observations cover different reporting periods.",
        )

    currency_a = _normalize_currency(left.currency)
    currency_b = _normalize_currency(right.currency)

    if currency_a is None or currency_b is None:
        return result(
            "insufficient_evidence",
            "A supported currency is required for both observations.",
        )

    if currency_a != currency_b:
        return result(
            "incomparable",
            "Currency differs; no currency conversion was attempted.",
        )

    unit_a = _normalize_unit(left.unit)
    unit_b = _normalize_unit(right.unit)

    if unit_a is None or unit_b is None:
        return result(
            "insufficient_evidence",
            "A supported unit is required for both observations.",
        )

    if (
        not isinstance(left.basis, str)
        or not left.basis.strip()
        or not isinstance(right.basis, str)
        or not right.basis.strip()
    ):
        return result(
            "insufficient_evidence",
            "Accounting or measurement basis is missing.",
        )

    if left.basis.strip().casefold() != right.basis.strip().casefold():
        return result(
            "incomparable",
            "Accounting or measurement basis differs.",
        )

    value_a = left.value * _UNIT_MULTIPLIERS[unit_a]
    value_b = right.value * _UNIT_MULTIPLIERS[unit_b]

    if not math.isfinite(value_a) or not math.isfinite(value_b):
        return result(
            "insufficient_evidence",
            "A normalized financial value is not finite.",
        )

    scale = max(abs(value_a), abs(value_b))

    difference_ratio = (
        0.0
        if scale == 0
        else abs(value_a - value_b) / scale
    )

    if difference_ratio > tolerance:
        return result(
            "discrepancy",
            "Same-field values differ beyond the configured tolerance; investigate the source evidence.",
            value_a,
            value_b,
            difference_ratio,
        )

    return result(
        "consistent",
        "Same-field values are within the configured tolerance.",
        value_a,
        value_b,
        difference_ratio,
    )
