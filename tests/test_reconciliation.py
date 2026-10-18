import pytest

from cross_check.reconciliation import (
    FinancialObservation,
    reconcile_observations,
)


def observation(
    source,
    value,
    *,
    period="FY2025",
    currency="INR",
    unit="crore",
    basis="reported_revenue",
    field="revenue",
    page=1,
):
    return FinancialObservation(
        field=field,
        value=value,
        source=source,
        period=period,
        currency=currency,
        unit=unit,
        basis=basis,
        page=page,
        evidence_text=f"{field}: {value}",
    )


def test_matching_values_are_consistent():
    results = reconcile_observations([
        observation("annual_report.pdf", 100),
        observation("gst_return.pdf", 102),
    ])

    assert len(results) == 1
    assert results[0].status == "consistent"
    assert results[0].source_a == "annual_report.pdf"
    assert results[0].page_b == 1


def test_material_difference_is_flagged():
    results = reconcile_observations([
        observation("annual_report.pdf", 100),
        observation("gst_return.pdf", 140),
    ])

    assert results[0].status == "discrepancy"
    assert results[0].difference_ratio == pytest.approx(40 / 140)
    assert results[0].value_a_normalized == 1_000_000_000
    assert results[0].value_b_normalized == 1_400_000_000


def test_equivalent_units_are_normalized_before_comparison():
    results = reconcile_observations([
        observation("annual_report.pdf", 1, unit="crore"),
        observation("ledger.csv", 100, unit="lakh"),
    ])

    assert results[0].status == "consistent"
    assert results[0].difference_ratio == pytest.approx(0.0)


def test_different_periods_are_incomparable():
    results = reconcile_observations([
        observation("annual_report.pdf", 100, period="FY2024"),
        observation("gst_return.pdf", 140, period="FY2025"),
    ])

    assert results[0].status == "incomparable"
    assert results[0].difference_ratio is None


def test_missing_period_is_insufficient_evidence():
    results = reconcile_observations([
        observation("annual_report.pdf", 100, period=None),
        observation("gst_return.pdf", 100),
    ])

    assert results[0].status == "insufficient_evidence"


def test_different_currency_is_not_silently_converted():
    results = reconcile_observations([
        observation("annual_report.pdf", 100, currency="INR"),
        observation("foreign_report.pdf", 100, currency="USD"),
    ])

    assert results[0].status == "incomparable"


def test_different_measurement_basis_is_incomparable():
    results = reconcile_observations([
        observation("annual_report.pdf", 100, basis="gross_revenue"),
        observation("gst_return.pdf", 100, basis="taxable_turnover"),
    ])

    assert results[0].status == "incomparable"


def test_different_fields_are_not_compared():
    results = reconcile_observations([
        observation("annual_report.pdf", 100, field="revenue"),
        observation("bank_statement.pdf", 100, field="bank_deposits"),
    ])

    assert results == []


def test_tolerance_is_validated():
    with pytest.raises(ValueError):
        reconcile_observations([], tolerance=1.5)


def test_zero_values_do_not_divide_by_zero():
    results = reconcile_observations([
        observation("annual_report.pdf", 0),
        observation("gst_return.pdf", 0),
    ])

    assert results[0].status == "consistent"
    assert results[0].difference_ratio == 0.0

import math


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_financial_values_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        reconcile_observations([
            observation("annual_report.pdf", value),
            observation("gst_return.pdf", 100),
        ])


@pytest.mark.parametrize("value", ["not-a-number", True])
def test_invalid_financial_values_are_rejected(value):
    with pytest.raises(ValueError, match="numeric"):
        reconcile_observations([
            observation("annual_report.pdf", value),
            observation("gst_return.pdf", 100),
        ])


@pytest.mark.parametrize("page", [0, -1, 1.5, True])
def test_invalid_page_numbers_are_rejected(page):
    with pytest.raises(ValueError, match="page"):
        reconcile_observations([
            observation("annual_report.pdf", 100, page=page),
            observation("gst_return.pdf", 100),
        ])


def test_blank_reporting_period_is_insufficient_evidence():
    results = reconcile_observations([
        observation("annual_report.pdf", 100, period="   "),
        observation("gst_return.pdf", 100),
    ])
    assert results[0].status == "insufficient_evidence"


def test_blank_measurement_basis_is_insufficient_evidence():
    results = reconcile_observations([
        observation("annual_report.pdf", 100, basis="   "),
        observation("gst_return.pdf", 100),
    ])
    assert results[0].status == "insufficient_evidence"


def test_non_finite_tolerance_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        reconcile_observations([], tolerance=math.nan)

def test_numeric_string_is_rejected_before_comparison():
    with pytest.raises(ValueError, match="int or float"):
        reconcile_observations([
            observation("annual_report.pdf", "100"),
            observation("gst_return.pdf", 100),
        ])
