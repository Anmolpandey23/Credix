
import json
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ai-brain"))

from ingestion.service import IngestionService  # noqa: E402
from main import AIBrainPipeline  # noqa: E402
from rag.vectorless import VectorlessRAGExtractor  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    """Write a JSON test fixture."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _assert_valid_score(score: object) -> None:
    """Assert that a score is a finite, non-boolean number in [0, 100]."""
    assert isinstance(score, (int, float)), (
        f"Expected a numeric score, got {type(score).__name__}"
    )
    assert not isinstance(score, bool), "Boolean values are not valid scores"
    assert math.isfinite(score), f"Expected a finite score, got {score}"
    assert 0 <= score <= 100, f"Expected score in [0, 100], got {score}"


def test_clean_company_no_flags(tmp_path: Path) -> None:
    file_path = tmp_path / "clean.json"
    _write_json(
        file_path,
        {
            "revenue": 1000,
            "net_profit": 120,
            "total_debt": 500,
            "bank_deposits": 850,
            "gst_revenue": 1000,
            "receivables": 300,
            "operating_cash_flow": 100,
            "note": "collections stable and no overdue payments",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Clean Corp", [file_path])

    assert result["company"] == "Clean Corp"
    assert result["risk_flags"] == []
    assert result["financials"]["revenue"] == 1000.0


def test_revenue_mismatch_flag(tmp_path: Path) -> None:
    file_path = tmp_path / "mismatch.json"
    _write_json(
        file_path,
        {
            "revenue": 1000,
            "net_profit": 100,
            "total_debt": 400,
            "bank_deposits": 500,
            "gst_revenue": 1000,
            "receivables": 200,
            "operating_cash_flow": 90,
            "risk_note": "tax demand received",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Mismatch Pvt", [file_path])

    risk_types = [flag["type"] for flag in result["risk_flags"]]
    assert "revenue_mismatch" in risk_types


def test_missing_data_insufficient(tmp_path: Path) -> None:
    file_path = tmp_path / "missing.json"
    _write_json(
        file_path,
        {
            "revenue": 1000,
            "gst_revenue": 1000,
            "note": "insolvency rumor in market",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Sparse Ltd", [file_path])

    assert result["status"] == "insufficient_data"
    assert result["confidence"] == 0.0
    assert result["confidence_level"] == "low"
    assert result["warnings"]


def test_high_debt_flag(tmp_path: Path) -> None:
    file_path = tmp_path / "debt.json"
    _write_json(
        file_path,
        {
            "revenue": 1000,
            "net_profit": 100,
            "total_debt": 900,
            "bank_deposits": 900,
            "gst_revenue": 1000,
            "receivables": 250,
            "operating_cash_flow": 80,
            "comment": "cash flow under stress",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Debt Heavy", [file_path])

    risk_types = [flag["type"] for flag in result["risk_flags"]]
    assert "high_leverage" in risk_types


def test_synonym_mapping_for_gst_revenue(tmp_path: Path) -> None:
    file_path = tmp_path / "synonyms.json"
    _write_json(
        file_path,
        {
            "Total Revenue": 200,
            "GST Declared Revenue": 100,
            "PAT": 40,
            "Bank Collections": 120,
            "Trade Receivables": 20,
            "Total Debt": 50,
            "Cash Flow From Operations": 35,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Synonym Works", [file_path])

    assert result["financials"]["gst_revenue"] == 100.0
    assert result["financials"]["bank_deposits"] == 120.0
    assert result["financials"]["net_profit"] == 40.0


def test_fallback_mapping_when_value_not_adjacent(tmp_path: Path) -> None:
    file_path = tmp_path / "fallback.json"
    _write_json(
        file_path,
        {
            "narrative": (
                "GST declared revenue for FY24 amounted to INR 150 "
                "after reconciliations."
            ),
            "Revenue": 150,
            "Net Profit": 30,
            "Bank Deposits": 130,
            "Receivables": 40,
            "Total Debt": 20,
            "Operating Cash Flow": 28,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Fallback Works", [file_path])

    assert result["financials"]["gst_revenue"] == 150.0


def test_approximate_range_values(tmp_path: Path) -> None:
    """Ranges should produce midpoints and retain extraction metadata."""
    file_path = tmp_path / "range.json"
    _write_json(
        file_path,
        {
            "Bank Inflows": "60-65 Cr",
            "GST Filings": "around 100 Cr",
            "Revenue": "~95-105 Cr",
            "Net Profit": "10 Cr",
            "Total Debt": "90 Cr",
            "Receivables": "70 Cr",
            "Operating Cash Flow": "2 Cr",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "Range Test", [file_path], include_metadata=True
    )

    assert result["financials"]["bank_deposits"] == 62.5
    assert result["financials"]["gst_revenue"] == 100.0
    assert result["financials"]["revenue"] == 100.0

    field_metrics = result.get("_metadata", {}).get("field_metrics", {})
    bank_metrics = field_metrics.get("bank_deposits", {})

    assert bank_metrics.get("extraction_type") in {
        "range",
        "approximate",
    }


def test_fuzzy_phrase_extraction(tmp_path: Path) -> None:
    """Fuzzy language should not prevent financial value extraction."""
    file_path = tmp_path / "fuzzy.json"
    _write_json(
        file_path,
        {
            "text": (
                "Cash flow from ops is weak around 2-3 Cr\n"
                "Bank inflows approximately 62 Cr\n"
                "GST turnover ~100 Cr (estimated)\n"
                "Revenue approx. 100 Cr\n"
                "Net Profit roughly 10 Cr\n"
                "Total Debt about 90 Cr\n"
                "Trade Receivables ca. 70 Cr"
            ),
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "Fuzzy Test", [file_path], include_metadata=True
    )

    financials = result["financials"]
    assert financials["operating_cash_flow"] == 2.5
    assert financials["bank_deposits"] == 62.0
    assert financials["gst_revenue"] == 100.0
    assert financials["revenue"] == 100.0

    field_metrics = result.get("_metadata", {}).get("field_metrics", {})
    ocf_metrics = field_metrics.get("operating_cash_flow", {})

    assert ocf_metrics.get("extraction_type") in {
        "approximate",
        "range",
    }


def test_new_synonyms_bank_deposits(tmp_path: Path) -> None:
    file_path = tmp_path / "bank_synonyms.json"
    _write_json(
        file_path,
        {
            "revenue": 1000,
            "net_profit": 100,
            "total_debt": 400,
            "cash_inflow": 550,
            "gst_revenue": 1000,
            "receivables": 200,
            "operating_cash_flow": 90,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Bank Synonyms", [file_path])

    assert result["financials"]["bank_deposits"] == 550.0


def test_new_synonyms_gst_revenue(tmp_path: Path) -> None:
    file_path = tmp_path / "gst_synonyms.json"
    _write_json(
        file_path,
        {
            "revenue": 1000,
            "net_profit": 100,
            "total_debt": 400,
            "bank_deposits": 500,
            "gst_filings": 1000,
            "receivables": 200,
            "operating_cash_flow": 90,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("GST Synonyms", [file_path])

    assert result["financials"]["gst_revenue"] == 1000.0


def test_new_synonyms_operating_cash_flow(tmp_path: Path) -> None:
    file_path = tmp_path / "ocf_synonyms.json"
    _write_json(
        file_path,
        {
            "revenue": 1000,
            "net_profit": 100,
            "total_debt": 400,
            "bank_deposits": 500,
            "gst_revenue": 1000,
            "receivables": 200,
            "cash_flow_from_ops": 90,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("OCF Synonyms", [file_path])

    assert result["financials"]["operating_cash_flow"] == 90.0


def test_confidence_low_when_approximate(tmp_path: Path) -> None:
    file_path = tmp_path / "confidence.json"
    _write_json(
        file_path,
        {
            "Revenue": "approximately 100-110 Cr",
            "Net Profit": 50,
            "Total Debt": "~500 Cr",
            "Bank Deposits": 750,
            "GST Revenue": "around 100 Cr",
            "Receivables": 300,
            "Operating Cash Flow": 100,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "Confidence Test", [file_path], include_metadata=True
    )

    field_metrics = result.get("_metadata", {}).get("field_metrics", {})

    for field in ("revenue", "total_debt", "gst_revenue"):
        assert field_metrics.get(field, {}).get("level") in {
            "low",
            "medium",
        }

    for field in ("net_profit", "bank_deposits"):
        assert field_metrics.get(field, {}).get("level") == "high"


def test_overall_confidence_reflects_extraction_quality(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "approx_data.json"
    _write_json(
        file_path,
        {
            "Revenue": "95-105 Cr",
            "Net Profit": "10 Cr",
            "Total Debt": "~50 Cr",
            "Bank Deposits": "100-120 Cr",
            "GST Revenue": "around 100 Cr",
            "Receivables": "30 Cr",
            "Operating Cash Flow": "8-12 Cr",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Approx Data Test", [file_path])

    assert isinstance(result.get("confidence"), (int, float))
    assert result["confidence_level"] in {"high", "medium", "low"}


def test_confidence_penalty_for_range_extraction(
    tmp_path: Path,
) -> None:
    """Range data should be extracted and marked as a range."""
    file_path = tmp_path / "ranges.json"
    _write_json(
        file_path,
        {
            "Revenue": "100-120 Cr",
            "Net Profit": 50,
            "Total Debt": 100,
            "Bank Deposits": 500,
            "GST Revenue": 100,
            "Receivables": 200,
            "Operating Cash Flow": 50,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "Range Test", [file_path], include_metadata=True
    )

    assert result["financials"]["revenue"] == 110.0

    metadata = result.get("_metadata", {})
    revenue_metrics = metadata.get("field_metrics", {}).get("revenue", {})
    assert revenue_metrics.get("extraction_type") == "range"
    assert revenue_metrics.get("level") in {"low", "medium"}


def test_critical_field_approximate_reduces_confidence(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "critical_approx.json"
    _write_json(
        file_path,
        {
            "Revenue": 1000,
            "Net Profit": 100,
            "Total Debt": 500,
            "Bank Deposits": "~850",
            "GST Revenue": "around 1000",
            "Receivables": 300,
            "Operating Cash Flow": 100,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "Critical Approx", [file_path], include_metadata=True
    )

    assert result["confidence"] == result["data_quality_score"]

    quality_artifacts = (
        result.get("_metadata", {}).get("quality_artifacts", {})
    )
    assert quality_artifacts.get("critical_approx") is True


def test_high_approximate_ratio_reduces_confidence(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "high_approx.json"
    _write_json(
        file_path,
        {
            "Revenue": "~100 Cr",
            "Net Profit": "around 10 Cr",
            "Total Debt": "~50 Cr",
            "Bank Deposits": 500,
            "GST Revenue": "approx 100 Cr",
            "Receivables": "~30 Cr",
            "Operating Cash Flow": "~5 Cr",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "High Approx", [file_path], include_metadata=True
    )

    assert result["confidence_level"] in {"medium", "low"}

    quality_artifacts = (
        result.get("_metadata", {}).get("quality_artifacts", {})
    )
    assert quality_artifacts.get("high_approximate_ratio") is True


def test_field_level_confidence_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "field_conf.json"
    _write_json(
        file_path,
        {
            "Revenue": 500,
            "Net Profit": "10-15",
            "Total Debt": "~100",
            "Bank Deposits": 750,
            "GST Revenue": "around 500",
            "Receivables": 200,
            "Operating Cash Flow": "5-8",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "Field Conf", [file_path], include_metadata=True
    )

    field_metrics = result.get("_metadata", {}).get("field_metrics", {})

    revenue_metrics = field_metrics.get("revenue", {})
    assert revenue_metrics.get("extraction_type") is not None
    assert revenue_metrics.get("level") is not None

    net_profit_metrics = field_metrics.get("net_profit", {})
    assert net_profit_metrics.get("extraction_type") in {
        "range",
        "approximate",
    }
    assert net_profit_metrics.get("level") in {"medium", "low"}

    bank_metrics = field_metrics.get("bank_deposits", {})
    assert bank_metrics.get("extraction_type") in {
        "exact",
        "approximate",
    }


def test_extraction_quality_detection(tmp_path: Path) -> None:
    clean_file = tmp_path / "clean.json"
    _write_json(
        clean_file,
        {
            "Revenue": 1000,
            "Net Profit": 100,
            "Total Debt": 500,
            "Bank Deposits": 850,
            "GST Revenue": 1000,
            "Receivables": 300,
            "Operating Cash Flow": 100,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "Clean Extraction", [clean_file], include_metadata=True
    )

    assert result["confidence_level"] == "high"
    assert result["confidence"] > 0.90


def test_missing_fields_penalty(tmp_path: Path) -> None:
    file_path = tmp_path / "missing.json"
    _write_json(
        file_path,
        {
            "Revenue": 1000,
            "Net Profit": 100,
            "Total Debt": 500,
            "GST Revenue": 1000,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Missing Fields", [file_path])

    assert result["confidence"] < 0.90


def test_ingestion_supports_directory_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "finance_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    sample_file = input_dir / "sample.json"
    _write_json(sample_file, {"revenue": 1000})

    service = IngestionService(storage_dir=tmp_path / "storage")
    stored = service.ingest_files([input_dir])

    assert len(stored) == 1
    assert stored[0].name == "sample.json"
    assert stored[0].exists()


def test_ingestion_infers_single_shorthand_file(
    tmp_path: Path,
) -> None:
    sample_pdf = tmp_path / "Finance .pdf"
    sample_pdf.write_bytes(b"%PDF-1.4\n%test\n")

    service = IngestionService(storage_dir=tmp_path / "storage")
    stored = service.ingest_files([tmp_path / "Finance"])

    assert len(stored) == 1
    assert stored[0].name == "Finance .pdf"


def test_structured_output_contains_risk_contributions_and_quality_score(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "structured.json"
    _write_json(
        file_path,
        {
            "Revenue": "98-102 Cr",
            "Net Profit": "9.5-10 Cr",
            "Total Debt": "85-95 Cr",
            "Bank Credits": "around 60 to 65 Cr",
            "GST Turnover": "100 Cr",
            "Trade Receivables": "65-75 Cr",
            "Cash Flow from Operations": "2-3 Cr",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Structured Co", [file_path])

    assert "risk_contributions" in result

    for flag in result["risk_flags"]:
        assert (
            result["risk_contributions"][flag["type"]]
            == flag["score_impact"]
        )

    score_map = {"high": 1.0, "medium": 0.75, "low": 0.5}
    field_confidence = result["field_confidence"]

    assert field_confidence
    expected = round(
        sum(score_map[level] for level in field_confidence.values())
        / len(field_confidence),
        2,
    )
    assert result["data_quality_score"] == expected


def test_company_name_extracted_from_document_content(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "company_name.json"
    _write_json(
        file_path,
        {
            "company_name": "ABC MANUFACTURING PVT LTD",
            "revenue": 100,
            "net_profit": 10,
            "total_debt": 80,
            "bank_deposits": 70,
            "gst_revenue": 100,
            "receivables": 40,
            "operating_cash_flow": 8,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Sample Company", [file_path])

    assert result["company"] == "ABC MANUFACTURING PVT LTD"


def test_stepx_approximate_value_preserved_without_modification(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "approx_preserve.json"
    _write_json(
        file_path,
        {
            "company_name": "ABC MANUFACTURING PVT LTD",
            "Revenue": "approx 102 Cr",
            "GST Turnover": "100 Cr",
            "Bank Credits": "around 70 Cr",
            "Net Profit": "20 Cr",
            "Total Debt": "95 Cr",
            "Trade Receivables": "75 Cr",
            "Cash Flow from Operations": "3.2 Cr",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "Sample Company", [file_path], include_metadata=True
    )

    assert result["financials"]["revenue"] == 102.0
    assert result["field_confidence"]["revenue"] == "low"


def test_stepx_range_overrides_approximate_for_same_field(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "range_override.json"
    _write_json(
        file_path,
        {
            "company_name": "ABC MANUFACTURING PVT LTD",
            "text": "Revenue ~100 Cr. Revenue: 98-102 Cr.",
            "GST Turnover": "100 Cr",
            "Bank Credits": "around 70 Cr",
            "Net Profit": "20 Cr",
            "Total Debt": "95 Cr",
            "Trade Receivables": "75 Cr",
            "Cash Flow from Operations": "3.2 Cr",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "Sample Company", [file_path], include_metadata=True
    )

    assert result["financials"]["revenue"] == 100.0
    assert result["field_confidence"]["revenue"] == "low"


def test_approx_indicator_before_value_marks_low_confidence(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "approx_before.json"
    _write_json(
        file_path,
        {
            "Revenue": "approx. 102 Cr",
            "Net Profit": 20,
            "Total Debt": 95,
            "Bank Deposits": 70,
            "GST Revenue": 100,
            "Receivables": 75,
            "Operating Cash Flow": 3.2,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Sample Company", [file_path])

    assert result["financials"]["revenue"] == 102.0
    assert result["field_confidence"]["revenue"] == "low"


def test_approx_indicator_after_value_marks_low_confidence(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "approx_after.json"
    _write_json(
        file_path,
        {
            "Revenue": "102 Cr (estimated)",
            "Net Profit": 20,
            "Total Debt": 95,
            "Bank Deposits": 70,
            "GST Revenue": 100,
            "Receivables": 75,
            "Operating Cash Flow": 3.2,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Sample Company", [file_path])

    assert result["financials"]["revenue"] == 102.0
    assert result["field_confidence"]["revenue"] == "low"


def test_range_priority_override_always_uses_midpoint(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "range_priority.json"
    _write_json(
        file_path,
        {
            "text": (
                "Total Borrowings / Debt: ~85 - 95 Cr. "
                "Cash Flow from Operations: ~2 to 3 Cr."
            ),
            "Revenue": 100,
            "Net Profit": 10,
            "Bank Deposits": 62,
            "GST Revenue": 100,
            "Trade Receivables": 70,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Sample Company", [file_path])

    assert result["financials"]["total_debt"] == 90.0
    assert result["financials"]["operating_cash_flow"] == 2.5


def test_range_with_units_between_bounds_uses_midpoint(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "range_units.json"
    _write_json(
        file_path,
        {
            "text": "Cash Flow from Operations: ~2 Cr to 3 Cr only.",
            "Revenue": 100,
            "Net Profit": 10,
            "Total Debt": 90,
            "Bank Deposits": 62,
            "GST Revenue": 100,
            "Trade Receivables": 70,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Sample Company", [file_path])

    assert result["financials"]["operating_cash_flow"] == 2.5


def test_range_on_following_line_after_label_uses_midpoint(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "range_next_line.json"
    _write_json(
        file_path,
        {
            "text": (
                "Total Borrowings / Debt:\n"
                "~85 - 95 Cr (incl. term loans)"
            ),
            "Revenue": 100,
            "Net Profit": 10,
            "Bank Deposits": 62,
            "GST Revenue": 100,
            "Trade Receivables": 70,
            "Cash Flow from Operations": 3,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Sample Company", [file_path])

    assert result["financials"]["total_debt"] == 90.0


def test_mixed_units_are_normalized_before_ratio_rules(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "mixed_units.json"
    _write_json(
        file_path,
        {
            "Revenue": "50 Cr",
            "GST Revenue": "500 million",
            "Bank Deposits": "42 Cr",
            "Net Profit": "4 Cr",
            "Total Debt": "20 Cr",
            "Receivables": "10 Cr",
            "Operating Cash Flow": "3 Cr",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Unit Normalization", [file_path])

    assert result["financials"]["revenue"] == 50.0
    assert result["financials"]["gst_revenue"] == 50.0


def test_negation_no_debt_maps_to_zero(tmp_path: Path) -> None:
    file_path = tmp_path / "negation_debt.json"
    _write_json(
        file_path,
        {
            "text": (
                "Revenue: 100\n"
                "No debt outstanding as on date\n"
                "GST Revenue: 100\n"
                "Bank Deposits: 90\n"
                "Net Profit: 10\n"
                "Receivables: 20\n"
                "Operating Cash Flow: 9"
            ),
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Negation Test", [file_path])

    assert result["financials"]["total_debt"] == 0.0
    assert "high_leverage" not in [
        flag["type"] for flag in result["risk_flags"]
    ]


def test_year_numbers_do_not_form_false_revenue_range(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "year_range_conflict.json"
    _write_json(
        file_path,
        {
            "text": (
                "Revenue FY23 FY24: 100\n"
                "GST Revenue: 100\n"
                "Bank Deposits: 90\n"
                "Net Profit: 10\n"
                "Total Debt: 30\n"
                "Receivables: 20\n"
                "Operating Cash Flow: 8"
            ),
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Year Test", [file_path])

    assert result["financials"]["revenue"] == 100.0


def test_parenthesized_negative_values_are_preserved(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "negative_values.json"
    _write_json(
        file_path,
        {
            "Revenue": 1000,
            "Net Profit": "(120)",
            "GST Revenue": 1000,
            "Bank Deposits": 850,
            "Total Debt": 300,
            "Receivables": 200,
            "Operating Cash Flow": "(80)",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Negative Test", [file_path])

    assert result["financials"]["net_profit"] == -120.0
    assert result["financials"]["operating_cash_flow"] == -80.0


def test_investing_cash_flow_not_misclassified_as_operating(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "cashflow_types.json"
    _write_json(
        file_path,
        {
            "Cash flow from investing": 200,
            "Cash flow from financing": -150,
            "Revenue": 1000,
            "GST Revenue": 1000,
            "Bank Deposits": 850,
            "Net Profit": 120,
            "Total Debt": 200,
            "Receivables": 100,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Cashflow Type Test", [file_path])

    assert result["financials"]["operating_cash_flow"] is None


def test_field_confidence_degrades_on_low_quality_text(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "low_quality.json"
    _write_json(
        file_path,
        {
            "text": (
                "Revenue : 1000\n"
                "| | | | |\n"
                "Net Profit : 100\n"
                "` ^ ~ { } [ ]\n"
                "GST Revenue : 1000\n"
                "Bank Deposits : 850\n"
                "Total Debt : 300\n"
                "Receivables : 200\n"
                "Operating Cash Flow : 90"
            ),
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "Low Quality", [file_path], include_metadata=True
    )

    extraction_quality = (
        result.get("_metadata", {})
        .get("quality_artifacts", {})
        .get("extraction_quality")
    )
    assert extraction_quality in {"medium", "low"}


def test_gst_revenue_does_not_inherit_turnover_range() -> None:
    """GST revenue must not inherit a neighboring turnover range."""
    raw_text = (
        "Total Turnover approx Rs. 98-102 Cr (rounded). "
        "GST filings show around 100 Cr."
    )

    financials, metadata = VectorlessRAGExtractor().extract_financials(
        raw_text
    )

    revenue = metadata["revenue"]
    gst_revenue = metadata["gst_revenue"]

    assert financials["revenue"] == 100.0
    assert revenue is not None
    assert revenue.has_range is True
    assert revenue.min_value == 98.0
    assert revenue.max_value == 102.0

    assert financials["gst_revenue"] == 100.0
    assert gst_revenue is not None
    assert gst_revenue.is_approximate is True
    assert gst_revenue.has_range is False
    assert gst_revenue.min_value is None
    assert gst_revenue.max_value is None


def test_total_liabilities_are_not_extracted_as_total_debt(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "total_liabilities_only.json"
    _write_json(
        file_path,
        {
            "text": (
                "Revenue: 100 Cr\n"
                "Total Liabilities: 90 Cr\n"
                "GST Revenue: 100 Cr\n"
                "Bank Deposits: 90 Cr\n"
                "Net Profit: 10 Cr\n"
                "Receivables: 20 Cr\n"
                "Operating Cash Flow: 8 Cr"
            ),
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Liabilities Test", [file_path])

    assert result["financials"]["total_debt"] is None


def test_safe_ratio_handles_invalid_and_negative_values(
    tmp_path: Path,
) -> None:
    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    ratio = pipeline._safe_ratio

    assert ratio(10, 2) == 5.0
    assert ratio(1, 3) == 0.33

    assert ratio(None, 2) is None
    assert ratio(10, None) is None
    assert ratio(10, 0) is None

    assert ratio(math.nan, 2) is None
    assert ratio(10, math.inf) is None
    assert ratio(-math.inf, 2) is None

    assert ratio(True, 2) is None
    assert ratio(10, False) is None

    assert ratio(-10, 2) == -5.0
    assert ratio(10, -2) == -5.0


def test_negative_operating_cash_flow_gets_dedicated_risk_flag(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "negative_operating_cash_flow.json"
    _write_json(
        file_path,
        {
            "Revenue": 1000,
            "Net Profit": "(120)",
            "GST Revenue": 1000,
            "Bank Deposits": 850,
            "Total Debt": 300,
            "Receivables": 200,
            "Operating Cash Flow": "(80)",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Negative Cash Flow Test", [file_path])

    assert result["financials"]["operating_cash_flow"] == -80.0

    flags = {flag["type"]: flag for flag in result["risk_flags"]}
    assert "negative_operating_cash_flow" in flags
    assert (
        flags["negative_operating_cash_flow"]["metric"]
        == "operating_cash_flow"
    )
    assert flags["negative_operating_cash_flow"]["value"] == -80.0


def test_pipeline_returns_independent_intellicredit_risk(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "integration_financials.json"
    _write_json(
        file_path,
        {
            "revenue": 1000,
            "net_profit": 120,
            "total_debt": 500,
            "bank_deposits": 850,
            "gst_revenue": 1000,
            "receivables": 300,
            "operating_cash_flow": 100,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Integration Test Corp", [file_path])

    assert result.get("status") != "failed"
    assert "final_score" in result
    assert "intellicredit_risk" in result

    risk = result["intellicredit_risk"]
    assert risk["status"] == "success"
    assert risk["engine"] == "IntelliCredit"
    assert risk["score_direction"] == "higher_score_means_higher_risk"
    assert risk["web_research_performed"] is False

    # The score systems have distinct meanings and must remain separate.
    _assert_valid_score(result["final_score"])
    _assert_valid_score(risk["final_score"])


def test_confidence_semantics_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "confidence_semantics.json"
    _write_json(
        file_path,
        {
            "Revenue": 100,
            "Net Profit": 10,
            "Total Debt": 50,
            "Bank Deposits": 80,
            "GST Revenue": 100,
            "Receivables": 20,
            "Operating Cash Flow": 8,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run(
        "Confidence Semantics Co",
        [file_path],
        include_metadata=True,
    )

    metadata = result["_metadata"]
    semantics = metadata["confidence_semantics"]

    assert "Extraction-level" in semantics["field_metrics"]
    assert "data_quality_score" in semantics["field_confidence"]
    assert "not probability" in semantics["data_quality_score"]
    assert "not probability" in semantics["confidence"]

    assert result["confidence"] == result["data_quality_score"]
    assert set(result["field_confidence"].values()) <= {
        "high",
        "medium",
        "low",
    }

    for key in (
        "field_metrics",
        "quality_artifacts",
        "field_details",
        "source_evidence",
        "reconciliation",
    ):
        assert key in metadata



def test_intellicredit_failure_does_not_break_ai_brain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from risk_analysis import intellicredit_adapter

    def fail_assessment(**kwargs):
        raise RuntimeError("simulated adapter failure")

    monkeypatch.setattr(
        intellicredit_adapter,
        "analyze_offline_risk",
        fail_assessment,
    )

    file_path = tmp_path / "failure_financials.json"
    _write_json(
        file_path,
        {
            "revenue": 1000,
            "net_profit": 120,
            "total_debt": 500,
            "bank_deposits": 850,
            "gst_revenue": 1000,
            "receivables": 300,
            "operating_cash_flow": 100,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Failure Test Corp", [file_path])

    # AI Brain must remain operational when IntelliCredit fails.
    assert result.get("status") != "failed"
    _assert_valid_score(result.get("final_score"))

    # The independent engine reports its own failure.
    risk = result["intellicredit_risk"]
    assert risk["status"] == "failed"
    assert risk["engine"] == "IntelliCredit"
    assert risk["web_research_performed"] is False

    # Do not expose raw exception details in the public result.
    assert "simulated adapter failure" not in str(risk)

    # AI Brain's own score and financial assessment remain available.
    assert isinstance(result.get("financials"), dict)
    assert result["financials"]
    assert isinstance(result.get("risk_flags"), list)

    assert any(
        "IntelliCredit assessment could not be completed" in warning
        and "AI Brain assessment is unchanged" in warning
        for warning in result["warnings"]
    )
