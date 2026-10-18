import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ai-brain"))

from main import AIBrainPipeline  # noqa: E402


def _write_financials(path: Path) -> None:
    path.write_text(
        json.dumps({
            "company": "Integration Test Ltd",
            "revenue": 1000,
            "net_profit": 120,
            "total_debt": 400,
            "bank_deposits": 850,
            "gst_revenue": 1000,
            "receivables": 200,
            "operating_cash_flow": 100,
        }),
        encoding="utf-8",
    )


def test_inspection_and_cam_are_connected_to_pipeline(tmp_path: Path) -> None:
    file_path = tmp_path / "financials.json"
    _write_financials(file_path)

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")

    baseline = pipeline.run("Integration Test Ltd", [file_path])
    inspected = pipeline.run(
        "Integration Test Ltd",
        [file_path],
        inspection_data={
            "capacity_utilization": 0.72,
            "machinery": "Operational; maintenance records reviewed",
            "workforce": 42,
            "expected_workforce": 45,
            "findings": [{
                "category": "safety",
                "severity": "low",
                "observation": "Safety register was available for review",
                "evidence": "Register presented during site visit",
            }],
        },
        external_signals=[{
            "type": "legal",
            "severity": "high",
            "confidence": 0.8,
            "source": "json",
            "evidence": "Analyst-supplied signal for integration testing",
        }],
    )

    assert "final_score" in inspected
    assert inspected["intellicredit_risk"]["status"] == "success"
    assert inspected["inspection"]["status"] == "validated"
    assert inspected["intellicredit_risk"]["status"] == "success"
    assert (
        inspected["intellicredit_risk"]["breakdown"]["operational"]
        != baseline["intellicredit_risk"]["breakdown"]["operational"]
    )

    memo = inspected["credit_appraisal_memo"]
    assert memo["status"] == "draft"
    assert memo["decision"] == "pending_human_review"
    assert memo["inspection_finding_count"] == 1
    assert memo["research_status"] == "provided_signals_not_live_researched"
    assert "Safety register was available for review" in memo["markdown"]

    # The two independent scores remain separate.
    assert "final_score" in inspected
    assert "final_score" in inspected["intellicredit_risk"]
    assert inspected["intellicredit_risk"]["web_research_performed"] is False


def test_invalid_inspection_is_rejected_safely(tmp_path: Path) -> None:
    file_path = tmp_path / "financials.json"
    _write_financials(file_path)

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "Integration Test Ltd",
        [file_path],
        inspection_data={"capacity_utilization": 150},
    )

    assert result["status"] == "failed"
    assert "Invalid inspection data" in result["reason"]


def test_cam_can_be_disabled(tmp_path: Path) -> None:
    file_path = tmp_path / "financials.json"
    _write_financials(file_path)

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "Integration Test Ltd",
        [file_path],
        include_cam=False,
    )

    assert "final_score" in result
    assert result["intellicredit_risk"]["status"] == "success"
    assert "credit_appraisal_memo" not in result


def test_reconciliation_and_cam_do_not_change_credit_assessment(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "score_isolation_financials.json"
    _write_financials(file_path)

    baseline_pipeline = AIBrainPipeline(
        storage_dir=tmp_path / "baseline_storage"
    )
    baseline = baseline_pipeline.run(
        "Score Isolation Ltd",
        [file_path],
        include_metadata=False,
        include_cam=False,
    )

    enriched_pipeline = AIBrainPipeline(
        storage_dir=tmp_path / "enriched_storage"
    )
    enriched = enriched_pipeline.run(
        "Score Isolation Ltd",
        [file_path],
        include_metadata=True,
        include_cam=True,
    )

    assert baseline["final_score"] == enriched["final_score"]
    assert baseline["financials"] == enriched["financials"]
    assert baseline["ratios"] == enriched["ratios"]
    assert baseline["risk_flags"] == enriched["risk_flags"]
    assert baseline["risk_contributions"] == enriched["risk_contributions"]
    assert baseline["data_quality_score"] == enriched["data_quality_score"]
    assert baseline["confidence"] == enriched["confidence"]
    assert (
        baseline["confidence_level"]
        == enriched["confidence_level"]
    )

    # IntelliCredit must also remain unchanged when metadata,
    # reconciliation, and CAM are enabled.
    baseline_ic = baseline["intellicredit_risk"]
    enriched_ic = enriched["intellicredit_risk"]

    assert baseline_ic["status"] == "success"
    assert enriched_ic["status"] == "success"

    for key in (
        "final_score",
        "score_audit",
        "risk_category",
        "breakdown",
        "missing_data_penalty",
        "reasons",
    ):
        assert baseline_ic[key] == enriched_ic[key], (
            f"IntelliCredit field changed when metadata/CAM "
            f"were enabled: {key}"
        )

    assert "reconciliation" not in baseline.get("_metadata", {})
    assert "reconciliation" in enriched["_metadata"]
    assert enriched["credit_appraisal_memo"]["status"] == "draft"
    assert (
        enriched["credit_appraisal_memo"]["decision"]
        == "pending_human_review"
    )
