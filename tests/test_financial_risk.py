import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ai-brain"))

from risk_analysis.intellicredit_engine.financial import (
    compute_debt_to_revenue,
    compute_receivables_to_revenue,
    compute_financial_risk,
    compute_profit_margin,
    weights,
)
from risk_analysis.intellicredit_engine.models import Financials  # noqa: E402


def test_new_financial_weights_sum_to_one():
    assert math.isclose(sum(weights.values()), 1.0)


def test_debt_to_revenue_requires_verified_comparability():
    financials = Financials(
        revenue=100,
        total_debt=90,
        debt_revenue_comparable=None,
    )
    assert compute_debt_to_revenue(financials) is None


def test_debt_to_revenue_returns_ratio_when_verified():
    financials = Financials(
        revenue=100,
        total_debt=90,
        debt_revenue_comparable=True,
    )
    assert compute_debt_to_revenue(financials) == 0.9


def test_receivables_to_revenue_returns_ratio_when_verified():
    financials = Financials(
        revenue=100,
        receivables=70,
        receivables_revenue_comparable=True,
    )
    assert compute_receivables_to_revenue(financials) == 0.7


def test_ratios_reject_zero_or_negative_revenue():
    for revenue in (0, -100):
        financials = Financials(
            revenue=revenue,
            total_debt=90,
            receivables=70,
            debt_revenue_comparable=True,
            receivables_revenue_comparable=True,
        )
        assert compute_debt_to_revenue(financials) is None
        assert compute_receivables_to_revenue(financials) is None


def test_ratios_reject_negative_amounts_and_non_finite_values():
    for amount in (-1, float("nan"), float("inf")):
        debt = Financials(
            revenue=100,
            total_debt=amount,
            debt_revenue_comparable=True,
        )
        receivables = Financials(
            revenue=100,
            receivables=amount,
            receivables_revenue_comparable=True,
        )
        assert compute_debt_to_revenue(debt) is None
        assert compute_receivables_to_revenue(receivables) is None


def test_unverified_ratios_are_reported_unavailable_not_as_fraud():
    financials = Financials(
        revenue=100,
        total_debt=90,
        receivables=70,
    )
    score, reasons = compute_financial_risk(financials)

    assert 0 <= score <= 100
    assert any("Debt-to-revenue ratio unavailable" in r for r in reasons)
    assert any("Receivables-to-revenue ratio unavailable" in r for r in reasons)
    assert not any("fraud" in r.lower() for r in reasons)


def test_verified_elevated_ratios_explain_credit_screening_concerns():
    financials = Financials(
        revenue=100,
        total_debt=120,
        receivables=70,
        debt_revenue_comparable=True,
        receivables_revenue_comparable=True,
    )
    _, reasons = compute_financial_risk(financials)

    assert any("leverage screening concern" in r for r in reasons)
    assert any("receivables aging" in r for r in reasons)

def test_source_evidence_does_not_enable_ratio_comparability():
    from risk_analysis.intellicredit_adapter import build_risk_input

    risk_input = build_risk_input(
        company_id="Evidence Regression Test",
        financials={
            "revenue": 100,
            "total_debt": 90,
            "receivables": 70,
        },
        field_metadata={
            "source_evidence": {
                "revenue": {
                    "source": "financial_report.pdf",
                    "page": 1,
                    "evidence_text": "Revenue approximately 100 Cr",
                    "status": "matched",
                },
                "total_debt": {
                    "source": "financial_report.pdf",
                    "page": 1,
                    "evidence_text": "Borrowings approximately 90 Cr",
                    "status": "matched",
                },
                "receivables": {
                    "source": "financial_report.pdf",
                    "page": 1,
                    "evidence_text": "Receivables approximately 70 Cr",
                    "status": "matched",
                },
            }
        },
    )

    mapped = risk_input["financials"]

    # Evidence availability must not imply accounting comparability.
    assert mapped["debt_revenue_comparable"] is None
    assert mapped["receivables_revenue_comparable"] is None

    # Evidence remains available for audit purposes.
    assert (
        risk_input["field_metadata"]["source_evidence"]["revenue"]["status"]
        == "matched"
    )


def test_score_audit_reproduces_existing_aggregation():
    from risk_analysis.intellicredit_engine.aggregator import (
        aggregate_scores,
        build_score_audit,
    )

    inputs = {
        "financial_score": 60,
        "fraud_score": 20,
        "external_score": 50,
        "operational_score": 40,
        "missing_penalty": 7,
    }

    expected = aggregate_scores(**inputs)
    audit = build_score_audit(**inputs)

    assert audit["final_score_after_clamping"] == round(expected, 2)
    assert audit["weighted_sum"] == 43.0
    assert audit["missing_data_penalty"] == 7.0
    assert audit["raw_score_before_clamping"] == 50.0
    assert audit["clamped"] is False

    assert audit["weighted_contributions"] == {
        "financial": 21.0,
        "fraud": 6.0,
        "external": 10.0,
        "operational": 6.0,
    }


def test_score_audit_reports_clamping():
    from risk_analysis.intellicredit_engine.aggregator import (
        build_score_audit,
    )

    audit = build_score_audit(
        financial_score=100,
        fraud_score=100,
        external_score=100,
        operational_score=100,
        missing_penalty=15,
    )

    assert audit["raw_score_before_clamping"] == 115.0
    assert audit["final_score_after_clamping"] == 100.0
    assert audit["clamped"] is True


def test_offline_assessment_returns_consistent_score_audit():
    from risk_analysis.intellicredit_adapter import analyze_offline_risk
    from risk_analysis.intellicredit_engine.aggregator import weights as score_weights

    result = analyze_offline_risk(
        company_id="Score Audit Integration Test",
        financials={
            "revenue": 1_000_000,
            "net_profit": 100_000,
            "total_debt": 300_000,
            "bank_deposits": 200_000,
            "gst_revenue": 950_000,
            "receivables": 150_000,
            "operating_cash_flow": 120_000,
        },
    )

    assert result["status"] == "success"
    assert result["engine"] == "IntelliCredit"
    assert result["score_direction"] == "higher_score_means_higher_risk"

    # The production response exposes the audit.
    audit = result["score_audit"]
    assert isinstance(audit, dict)

    # The audit and the production result agree.
    assert (
        audit["final_score_after_clamping"]
        == result["final_score"]
    )
    assert 0 <= result["final_score"] <= 100

    # The audit describes the actual configured aggregation weights.
    assert audit["weights"] == score_weights
    assert math.isclose(sum(audit["weights"].values()), 1.0)

    # Verify the score arithmetic.
    assert math.isclose(
        audit["raw_score_before_clamping"],
        audit["weighted_sum"] + audit["missing_data_penalty"],
        abs_tol=1e-6,
    )

    # Existing response fields remain available.
    assert isinstance(result["breakdown"], dict)
    assert set(result["breakdown"]) == {
        "financial",
        "fraud",
        "external",
        "operational",
    }
    assert result["risk_category"] in {
        "Low Risk",
        "Medium Risk",
        "High Risk",
    }
    assert isinstance(result["evidence"], dict)
    assert result["web_research_performed"] is False

def test_fractional_scores_keep_audit_arithmetic_consistent():
    from risk_analysis.intellicredit_engine.aggregator import (
        aggregate_scores,
        build_score_audit,
    )

    scores = (33.333333, 47.123456, 62.987654, 21.456789)
    penalty = 3.0

    audit = build_score_audit(*scores, penalty)
    final_score = aggregate_scores(*scores, penalty)

    assert math.isclose(
        audit["weighted_sum"],
        sum(
            scores[i] * audit["weights"][name]
            for i, name in enumerate(
                ("financial", "fraud", "external", "operational")
            )
        ),
        abs_tol=1e-5,
    )
    assert math.isclose(
        audit["raw_score_before_clamping"],
        audit["weighted_sum"] + penalty,
        abs_tol=1e-5,
    )
    assert audit["final_score_after_clamping"] == round(final_score, 2)



def test_aggregate_score_clamps_lower_and_upper_boundaries():
    from risk_analysis.intellicredit_engine.aggregator import (
        aggregate_scores,
        build_score_audit,
    )

    # Valid lower and upper score boundaries.
    assert aggregate_scores(0, 0, 0, 0, 0) == 0
    assert aggregate_scores(100, 100, 100, 100, 0) == 100

    # A non-negative penalty can push the raw score above 100.
    high_audit = build_score_audit(100, 100, 100, 100, 15)

    assert high_audit["raw_score_before_clamping"] == 115
    assert high_audit["final_score_after_clamping"] == 100
    assert high_audit["clamped"] is True

    # Negative penalties must be rejected.
    import pytest

    with pytest.raises(ValueError, match="missing_penalty"):
        build_score_audit(0, 0, 0, 0, -10)


def test_aggregate_scores_rejects_module_scores_outside_range():
    import pytest
    from risk_analysis.intellicredit_engine.aggregator import aggregate_scores

    invalid_scores = [
        (-1, 50, 50, 50, 0),
        (101, 50, 50, 50, 0),
    ]

    for scores in invalid_scores:
        with pytest.raises(ValueError):
            aggregate_scores(*scores)


def test_aggregate_scores_rejects_non_finite_values():
    import math
    import pytest
    from risk_analysis.intellicredit_engine.aggregator import aggregate_scores

    invalid_values = [math.nan, math.inf, -math.inf]

    for value in invalid_values:
        with pytest.raises(ValueError):
            aggregate_scores(value, 50, 50, 50, 0)


def test_aggregate_scores_rejects_negative_missing_penalty():
    import pytest
    from risk_analysis.intellicredit_engine.aggregator import aggregate_scores

    with pytest.raises(ValueError):
        aggregate_scores(50, 50, 50, 50, -1)


def test_aggregate_scores_rejects_non_finite_penalty():
    import math
    import pytest
    from risk_analysis.intellicredit_engine.aggregator import aggregate_scores

    for penalty in [math.nan, math.inf, -math.inf]:
        with pytest.raises(ValueError):
            aggregate_scores(50, 50, 50, 50, penalty)


def test_profit_margin_requires_verified_comparability():
    financials = Financials(
        profit=10,
        gst_revenue=100,
        profit_revenue_comparable=None,
    )
    assert compute_profit_margin(financials) is None


def test_profit_margin_returns_ratio_when_verified():
    financials = Financials(
        profit=10,
        gst_revenue=100,
        profit_revenue_comparable=True,
    )
    assert compute_profit_margin(financials) == 0.1


def test_profit_margin_rejects_invalid_revenue():
    for revenue in (0, -100, float("nan"), float("inf")):
        financials = Financials(
            profit=10,
            gst_revenue=revenue,
            profit_revenue_comparable=True,
        )
        assert compute_profit_margin(financials) is None