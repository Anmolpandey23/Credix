import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ai-brain"))

from risk_analysis.intellicredit_engine.validator import (  # noqa: E402
    validate_and_standardize,
)


def _valid_input(signals):
    return {
        "company_id": "Validator Test Ltd",
        "financials": {},
        "operational_data": {},
        "external_signals": signals,
    }


def test_json_signal_preserves_source_and_evidence():
    result = validate_and_standardize(
        _valid_input([{
            "type": "legal",
            "severity": "high",
            "confidence": 0.8,
            "source": "json",
            "evidence": "Analyst-supplied legal-risk observation",
        }])
    )

    assert result.status == "ok"
    signal = result.data.external_signals[0]
    assert signal.source == "json"
    assert signal.evidence == "Analyst-supplied legal-risk observation"


def test_web_signal_preserves_source_and_evidence():
    result = validate_and_standardize(
        _valid_input([{
            "type": "legal",
            "severity": "high",
            "confidence": 0.9,
            "source": "web",
            "evidence": "A source-reported legal filing",
        }])
    )

    assert result.status == "ok"
    signal = result.data.external_signals[0]
    assert signal.source == "web"
    assert signal.evidence == "A source-reported legal filing"


def test_missing_source_defaults_to_json_for_compatibility():
    result = validate_and_standardize(
        _valid_input([{
            "type": "operational",
            "severity": "medium",
            "confidence": 0.5,
            "evidence": "Supplied by an analyst",
        }])
    )

    assert result.status == "ok"
    signal = result.data.external_signals[0]
    assert signal.source == "json"
    assert signal.evidence == "Supplied by an analyst"


def test_invalid_source_is_skipped_without_failing_validation():
    result = validate_and_standardize(
        _valid_input([
            {
                "type": "legal",
                "severity": "high",
                "confidence": 0.8,
                "source": "unverified",
                "evidence": "Invalid source label",
            },
            {
                "type": "operational",
                "severity": "low",
                "confidence": 0.4,
                "source": "json",
                "evidence": "Valid signal",
            },
        ])
    )

    assert result.status == "ok"
    assert len(result.data.external_signals) == 1
    assert result.data.external_signals[0].type == "operational"
    assert result.data.external_signals[0].source == "json"


def test_malformed_signal_is_skipped_without_failing_validation():
    result = validate_and_standardize(
        _valid_input([
            {"type": "legal"},
            {
                "type": "fraud",
                "severity": "critical",
                "confidence": 0.95,
                "source": "web",
                "evidence": "Valid evidence",
            },
        ])
    )

    assert result.status == "ok"
    assert len(result.data.external_signals) == 1
    assert result.data.external_signals[0].type == "fraud"
    assert result.data.external_signals[0].source == "web"


def test_empty_external_signals_do_not_trigger_zero_confidence_warning():
    result = validate_and_standardize(_valid_input([]))

    assert result.status == "ok"
    assert not any(
        "All signals have zero confidence" in issue
        for issue in result.warnings
    )


def test_all_zero_confidence_external_signals_trigger_warning():
    result = validate_and_standardize(
        _valid_input([
            {
                "type": "legal",
                "severity": "medium",
                "confidence": 0.0,
                "source": "json",
                "evidence": "Analyst-supplied observation",
            }
        ])
    )

    assert result.status == "ok"
    assert any(
        "All signals have zero confidence" in issue
        for issue in result.warnings
    )
