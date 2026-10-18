from risk_analysis.intellicredit_engine.fraud import (
    compute_fraud_risk,
)
from risk_analysis.intellicredit_engine.models import Financials


def test_missing_data_does_not_create_fraud_risk_points():
    score, reasons = compute_fraud_risk(Financials())

    assert score == 50.0  # Neutral: no evaluable evidence
    assert any("neutral" in reason.lower() for reason in reasons)


def test_unverified_gst_bank_comparison_is_not_scored():
    financials = Financials(
        gst_revenue=100,
        bank_deposits=20,
        gst_bank_comparable=None,
    )

    score, reasons = compute_fraud_risk(financials)

    assert score == 50.0
    assert any("GST-to-bank comparison not scored" in r for r in reasons)


def test_verified_gst_bank_discrepancy_is_scored():
    financials = Financials(
        gst_revenue=100,
        bank_deposits=20,
        gst_bank_comparable=True,
    )

    score, reasons = compute_fraud_risk(financials)

    assert score == 85.0
    assert any("investigate reporting periods" in r for r in reasons)


def test_missing_revenue_history_does_not_increase_score():
    financials = Financials(
        revenue_history=None,
        revenue_history_comparable=True,
    )

    score, reasons = compute_fraud_risk(financials)

    assert score == 50.0
    assert any("Revenue-spike signal unavailable" in r for r in reasons)


def test_verified_profit_cashflow_gap_is_an_investigation_signal():
    financials = Financials(
        profit=10,
        cash_flow=2,
        profit_cashflow_comparable=True,
    )

    score, reasons = compute_fraud_risk(financials)

    assert score == 80.0
    assert any("investigate earnings-to-cash conversion" in r for r in reasons)


def test_comparable_normal_gst_bank_ratio_gets_low_signal_score():
    financials = Financials(
        gst_revenue=100,
        bank_deposits=80,
        gst_bank_comparable=True,
    )

    score, _ = compute_fraud_risk(financials)

    assert score == 10.0
