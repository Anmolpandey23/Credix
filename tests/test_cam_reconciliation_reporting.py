import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ai-brain"))

from risk_analysis.intellicredit_engine.cam import (  # noqa: E402
    generate_credit_appraisal_memo,
)


def _generate_memo(reconciliation):
    return generate_credit_appraisal_memo(
        company="Reconciliation Test Ltd",
        financials={"revenue": 100.0},
        reconciliation=reconciliation,
    )


def test_zero_comparisons_does_not_imply_consistency():
    memo = _generate_memo({
        "status": "insufficient_evidence",
        "candidate_count": 7,
        "observation_count": 7,
        "comparison_count": 0,
        "comparisons": [],
    })

    assert (
        "No same-field observation pairs were available to compare"
        in memo["markdown"]
    )
    assert "cross-source consistency has not been established" in memo["markdown"]
    assert "No unresolved comparisons reported." not in memo["markdown"]


def test_completed_comparisons_report_no_unresolved_pairs_accurately():
    memo = _generate_memo({
        "status": "completed",
        "comparison_count": 3,
        "comparisons": [
            {"field": "revenue", "status": "consistent", "reason": "Within tolerance"},
            {"field": "net_profit", "status": "consistent", "reason": "Within tolerance"},
            {"field": "total_debt", "status": "consistent", "reason": "Within tolerance"},
        ],
    })

    assert (
        "No comparisons requiring additional evidence were reported "
        "among 3 comparison(s)."
    ) in memo["markdown"]


def test_unresolved_comparisons_are_still_listed():
    memo = _generate_memo({
        "status": "insufficient_evidence",
        "comparison_count": 1,
        "comparisons": [{
            "field": "revenue",
            "status": "insufficient_evidence",
            "reason": "Reporting period is missing.",
            "source_a": "annual_report.pdf",
            "source_b": "bank_statement.pdf",
        }],
    })

    assert "revenue: Reporting period is missing." in memo["markdown"]
    assert memo["unresolved_comparison_count"] == 1
