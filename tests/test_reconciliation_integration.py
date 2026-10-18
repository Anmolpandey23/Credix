import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ai-brain"))

from cross_check.reconciliation import (  # noqa: E402
    FinancialObservation,
    reconcile_observations,
)


def test_unknown_context_never_produces_false_consistency_or_discrepancy():
    observations = [
        FinancialObservation(
            field="revenue",
            value=100.0,
            source="annual_report.pdf",
            period=None,
            currency=None,
            unit=None,
            basis=None,
            page=3,
            evidence_text="Revenue: 100",
        ),
        FinancialObservation(
            field="revenue",
            value=150.0,
            source="gst_return.pdf",
            period=None,
            currency=None,
            unit=None,
            basis=None,
            page=2,
            evidence_text="Revenue: 150",
        ),
    ]

    results = reconcile_observations(observations)

    assert len(results) == 1
    assert results[0].status == "insufficient_evidence"
    assert results[0].difference_ratio is None
    assert results[0].source_a == "annual_report.pdf"
    assert results[0].source_b == "gst_return.pdf"
    assert results[0].page_a == 3
    assert results[0].page_b == 2
