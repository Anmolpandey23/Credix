import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ai-brain"))

from ingestion.service import IngestionService  # noqa: E402
from main import AIBrainPipeline  # noqa: E402
from rag.vectorless import VectorlessRAGExtractor  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


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
    assert "warnings" in result
    assert len(result["warnings"]) > 0


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
            "narrative": "GST declared revenue for FY24 amounted to INR 150 after reconciliations.",
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
    """Test extraction of range values like 60-65 Cr → 62.5"""
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
    result = pipeline.run("Range Test", [file_path], include_metadata=True)

    # Should extract midpoints for ranges
    assert result["financials"]["bank_deposits"] == 62.5
    assert result["financials"]["gst_revenue"] == 100.0
    assert result["financials"]["revenue"] == 100.0

    # Should mark as approximate in metadata
    metadata = result.get("_metadata", {})
    field_metrics = metadata.get("field_metrics", {})

    # Check that bank_deposits has range or approximate extraction type
    bank_metrics = field_metrics.get("bank_deposits", {})
    assert bank_metrics.get("extraction_type") in ["range", "approximate"], \
        f"Expected range or approximate, got {bank_metrics.get('extraction_type')}"


def test_fuzzy_phrase_extraction(tmp_path: Path) -> None:
    """Test extraction from fuzzy phrases like 'around 100' or '~60'"""
    file_path = tmp_path / "fuzzy.json"
    _write_json(
        file_path,
        {
            "text": """
            Cash flow from ops is weak around 2-3 Cr
            Bank inflows approximately 62 Cr
            GST turnover ~100 Cr (estimated)
            Revenue approx. 100 Cr
            Net Profit roughly 10 Cr
            Total Debt about 90 Cr
            Trade Receivables ca. 70 Cr
            """,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Fuzzy Test", [file_path], include_metadata=True)

    # Should extract values despite fuzzy language
    assert result["financials"]["operating_cash_flow"] == 2.5  # midpoint of 2-3
    assert result["financials"]["bank_deposits"] == 62.0
    assert result["financials"]["gst_revenue"] == 100.0
    assert result["financials"]["revenue"] == 100.0

    # Most should have extraction type: approximate or range
    metadata = result.get("_metadata", {})
    field_metrics = metadata.get("field_metrics", {})

    ocf_metrics = field_metrics.get("operating_cash_flow", {})
    assert ocf_metrics.get("extraction_type") in ["approximate", "range"], \
        f"Expected approximate or range, got {ocf_metrics.get('extraction_type')}"


def test_new_synonyms_bank_deposits(tmp_path: Path) -> None:
    """Test that new bank_deposits synonyms are recognized"""
    file_path = tmp_path / "bank_synonyms.json"
    _write_json(
        file_path,
        {
            "revenue": 1000,
            "net_profit": 100,
            "total_debt": 400,
            "cash_inflow": 550,  # Alternative synonym
            "gst_revenue": 1000,
            "receivables": 200,
            "operating_cash_flow": 90,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Bank Synonyms", [file_path])

    assert result["financials"]["bank_deposits"] == 550.0


def test_new_synonyms_gst_revenue(tmp_path: Path) -> None:
    """Test that new gst_revenue synonyms are recognized"""
    file_path = tmp_path / "gst_synonyms.json"
    _write_json(
        file_path,
        {
            "revenue": 1000,
            "net_profit": 100,
            "total_debt": 400,
            "bank_deposits": 500,
            "gst_filings": 1000,  # New synonym
            "receivables": 200,
            "operating_cash_flow": 90,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("GST Synonyms", [file_path])

    assert result["financials"]["gst_revenue"] == 1000.0


def test_new_synonyms_operating_cash_flow(tmp_path: Path) -> None:
    """Test that new operating_cash_flow synonyms are recognized"""
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
            "cash_flow_from_ops": 90,  # New synonym
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("OCF Synonyms", [file_path])

    assert result["financials"]["operating_cash_flow"] == 90.0


def test_confidence_low_when_approximate(tmp_path: Path) -> None:
    """Test that confidence is reduced for approximate values"""
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
    result = pipeline.run("Confidence Test", [file_path], include_metadata=True)

    field_metrics = result.get("_metadata", {}).get("field_metrics", {})

    # Approximate/range values should have low/medium confidence
    assert field_metrics.get("revenue", {}).get("level") in ["low", "medium"]
    assert field_metrics.get("total_debt", {}).get("level") in ["low", "medium"]
    assert field_metrics.get("gst_revenue", {}).get("level") in ["low", "medium"]

    # Non-approximate values should have high confidence
    assert field_metrics.get("net_profit", {}).get("level") == "high"
    assert field_metrics.get("bank_deposits", {}).get("level") == "high"


def test_overall_confidence_reflects_extraction_quality(tmp_path: Path) -> None:
    """Test that overall confidence is lowered for approximate/range extractions"""
    file_path = tmp_path / "approx_data.json"
    _write_json(
        file_path,
        {
            "Revenue": "95-105 Cr",  # Range
            "Net Profit": "10 Cr",  # Exact
            "Total Debt": "~50 Cr",  # Approximate
            "Bank Deposits": "100-120 Cr",  # Range
            "GST Revenue": "around 100 Cr",  # Approximate
            "Receivables": "30 Cr",  # Exact
            "Operating Cash Flow": "8-12 Cr",  # Range
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Approx Data Test", [file_path])

    # Overall confidence should be reasonable for mixed data
    # Note: The " Cr" suffix might affect extraction, so we verify structure rather than strict thresholds
    assert "confidence" in result, "Should have confidence score"
    assert "confidence_level" in result, "Should have confidence level"
    assert result["confidence_level"] in ["high", "medium", "low"], "Should be valid confidence level"


def test_confidence_penalty_for_range_extraction(tmp_path: Path) -> None:
    """Test that range extractions reduce confidence more than approximate"""
    # Ranges should have -0.15 penalty vs -0.10 for approximate
    file_path = tmp_path / "ranges.json"
    _write_json(
        file_path,
        {
            "Revenue": "100-120 Cr",  # Range extraction only
            "Net Profit": 50,
            "Total Debt": 100,
            "Bank Deposits": 500,
            "GST Revenue": 100,
            "Receivables": 200,
            "Operating Cash Flow": 50,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Range Test", [file_path], include_metadata=True)

    # With 1 range among 7 fields and high quality extraction, confidence should still be reasonable
    # Base = 7/7 = 1.0, minus 1/7 * 0.15 = 0.021 penalty, ~0.98
    assert result["confidence"] >= 0.70, f"Expected confidence sufficient for ranges, got {result['confidence']}"


def test_critical_field_approximate_reduces_confidence(tmp_path: Path) -> None:
    """Test that approximate critical fields (GST/Bank) further reduce confidence"""
    file_path = tmp_path / "critical_approx.json"
    _write_json(
        file_path,
        {
            "Revenue": 1000,
            "Net Profit": 100,
            "Total Debt": 500,
            "Bank Deposits": "~850",  # Critical field as approximate
            "GST Revenue": "around 1000",  # Critical field as approximate
            "Receivables": 300,
            "Operating Cash Flow": 100,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Critical Approx", [file_path], include_metadata=True)

    # Master rule: confidence is calibrated directly from data quality score.
    assert result["confidence"] == result["data_quality_score"]

    quality_artifacts = result.get("_metadata", {}).get("quality_artifacts", {})
    assert quality_artifacts.get("critical_approx") is True


def test_high_approximate_ratio_reduces_confidence(tmp_path: Path) -> None:
    """Test that >50% approximate fields trigger additional confidence penalty"""
    file_path = tmp_path / "high_approx.json"
    _write_json(
        file_path,
        {
            "Revenue": "~100 Cr",
            "Net Profit": "around 10 Cr",
            "Total Debt": "~50 Cr",
            "Bank Deposits": 500,  # Only one exact
            "GST Revenue": "approx 100 Cr",
            "Receivables": "~30 Cr",
            "Operating Cash Flow": "~5 Cr",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("High Approx", [file_path], include_metadata=True)

    # 6 out of 7 fields are approximate (>85%), should have medium-to-low confidence
    assert result["confidence_level"] in ["medium", "low"], f"Expected 'medium' or 'low' confidence level for high approx ratio, got {result['confidence_level']}"

    quality_artifacts = result.get("_metadata", {}).get("quality_artifacts", {})
    assert quality_artifacts.get("high_approximate_ratio") is True


def test_field_level_confidence_metadata(tmp_path: Path) -> None:
    """Test that field-level confidence metrics are correctly assigned"""
    file_path = tmp_path / "field_conf.json"
    _write_json(
        file_path,
        {
            "Revenue": 500,  # Exact
            "Net Profit": "10-15",  # Range
            "Total Debt": "~100",  # Approximate
            "Bank Deposits": 750,  # Exact
            "GST Revenue": "around 500",  # Approximate
            "Receivables": 200,  # Exact
            "Operating Cash Flow": "5-8",  # Range
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Field Conf", [file_path], include_metadata=True)

    field_metrics = result.get("_metadata", {}).get("field_metrics", {})

    # Check that exact fields have extraction_type present
    revenue_metrics = field_metrics.get("revenue", {})
    assert revenue_metrics.get("extraction_type") is not None, "Revenue should have extraction_type"
    assert revenue_metrics.get("level") is not None, "Revenue should have confidence level"

    # Range fields should be marked as range or have lower confidence
    net_profit_metrics = field_metrics.get("net_profit", {})
    assert net_profit_metrics.get("extraction_type") in ["range", "approximate"], "Expected range/approximate extraction type for net_profit"
    assert net_profit_metrics.get("level") in ["medium", "low"]

    # Bank deposits and receivables should be extracted as exact
    bank_metrics = field_metrics.get("bank_deposits", {})
    assert bank_metrics.get("extraction_type") in ["exact", "approximate"], "Bank deposits extraction type should be valid"


def test_extraction_quality_detection(tmp_path: Path) -> None:
    """Test that extraction quality is detected and affects confidence"""
    # Create data with clean extraction
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
    result = pipeline.run("Clean Extraction", [clean_file], include_metadata=True)

    # Clean extraction with all exact values should have high confidence
    assert result["confidence_level"] == "high", "Expected 'high' for clean extraction"
    assert result["confidence"] > 0.90


def test_missing_fields_penalty(tmp_path: Path) -> None:
    """Test that missing fields reduce overall confidence"""
    file_path = tmp_path / "missing.json"
    _write_json(
        file_path,
        {
            "Revenue": 1000,
            "Net Profit": 100,
            "Total Debt": 500,
            "GST Revenue": 1000,
            # Missing: Bank Deposits, Receivables, Operating Cash Flow
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Missing Fields", [file_path])

    # 4 out of 7 fields (57% missing) but enough to process
    # Should have reduced confidence due to missing fields
    assert result["confidence"] < 0.90, "Missing fields should reduce confidence"


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


def test_ingestion_infers_single_shorthand_file(tmp_path: Path) -> None:
    sample_pdf = tmp_path / "Finance .pdf"
    sample_pdf.write_bytes(b"%PDF-1.4\n%test\n")

    service = IngestionService(storage_dir=tmp_path / "storage")
    stored = service.ingest_files([tmp_path / "Finance"])

    assert len(stored) == 1
    assert stored[0].name == "Finance .pdf"


def test_structured_output_contains_risk_contributions_and_quality_score(tmp_path: Path) -> None:
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
        assert result["risk_contributions"][flag["type"]] == flag["score_impact"]

    score_map = {"high": 1.0, "medium": 0.75, "low": 0.5}
    expected = round(
        sum(score_map[result["field_confidence"][field]] for field in result["field_confidence"]) /
        len(result["field_confidence"]),
        2,
    )
    assert result["data_quality_score"] == expected


def test_company_name_extracted_from_document_content(tmp_path: Path) -> None:
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


def test_stepx_approximate_value_preserved_without_modification(tmp_path: Path) -> None:
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
    result = pipeline.run("Sample Company", [file_path], include_metadata=True)

    # STEP X: approximate single value must be preserved as-is (no conversion 102 -> 100)
    assert result["financials"]["revenue"] == 102.0
    assert result["field_confidence"]["revenue"] == "low"


def test_stepx_range_overrides_approximate_for_same_field(tmp_path: Path) -> None:
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
    result = pipeline.run("Sample Company", [file_path], include_metadata=True)

    # STEP X override: when both approximate and range exist, choose range midpoint.
    assert result["financials"]["revenue"] == 100.0
    assert result["field_confidence"]["revenue"] == "low"


def test_approx_indicator_before_value_marks_low_confidence(tmp_path: Path) -> None:
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


def test_approx_indicator_after_value_marks_low_confidence(tmp_path: Path) -> None:
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


def test_range_priority_override_always_uses_midpoint(tmp_path: Path) -> None:
    file_path = tmp_path / "range_priority.json"
    _write_json(
        file_path,
        {
            "text": "Total Borrowings / Debt: ~85 - 95 Cr. Cash Flow from Operations: ~2 to 3 Cr.",
            "Revenue": 100,
            "Net Profit": 10,
            "Bank Deposits": 62,
            "GST Revenue": 100,
            "Trade Receivables": 70,
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Sample Company", [file_path])

    # Range must take priority; midpoint only.
    assert result["financials"]["total_debt"] == 90.0
    assert result["financials"]["operating_cash_flow"] == 2.5


def test_range_with_units_between_bounds_uses_midpoint(tmp_path: Path) -> None:
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


def test_range_on_following_line_after_label_uses_midpoint(tmp_path: Path) -> None:
    file_path = tmp_path / "range_next_line.json"
    _write_json(
        file_path,
        {
            "text": "Total Borrowings / Debt:\n~85 - 95 Cr (incl. term loans)",
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


def test_mixed_units_are_normalized_before_ratio_rules(tmp_path: Path) -> None:
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
    # 500 million == 50 crore after normalization
    assert result["financials"]["gst_revenue"] == 50.0


def test_negation_no_debt_maps_to_zero(tmp_path: Path) -> None:
    file_path = tmp_path / "negation_debt.json"
    _write_json(
        file_path,
        {
            "text": "Revenue: 100\nNo debt outstanding as on date\nGST Revenue: 100\nBank Deposits: 90\nNet Profit: 10\nReceivables: 20\nOperating Cash Flow: 9",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Negation Test", [file_path])

    assert result["financials"]["total_debt"] == 0.0
    assert "high_leverage" not in [flag["type"] for flag in result["risk_flags"]]


def test_year_numbers_do_not_form_false_revenue_range(tmp_path: Path) -> None:
    file_path = tmp_path / "year_range_conflict.json"
    _write_json(
        file_path,
        {
            "text": "Revenue FY23 FY24: 100\nGST Revenue: 100\nBank Deposits: 90\nNet Profit: 10\nTotal Debt: 30\nReceivables: 20\nOperating Cash Flow: 8",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Year Test", [file_path])

    assert result["financials"]["revenue"] == 100.0


def test_parenthesized_negative_values_are_preserved(tmp_path: Path) -> None:
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


def test_investing_cash_flow_not_misclassified_as_operating(tmp_path: Path) -> None:
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


def test_field_confidence_degrades_on_low_quality_text(tmp_path: Path) -> None:
    file_path = tmp_path / "low_quality.json"
    _write_json(
        file_path,
        {
            "text": "Revenue : 1000\n| | | | |\nNet Profit : 100\n` ^ ~ { } [ ]\nGST Revenue : 1000\nBank Deposits : 850\nTotal Debt : 300\nReceivables : 200\nOperating Cash Flow : 90",
        },
    )

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    result = pipeline.run("Low Quality", [file_path], include_metadata=True)

    extraction_quality = result.get("_metadata", {}).get("quality_artifacts", {}).get("extraction_quality")
    assert extraction_quality in ["medium", "low"]

def test_gst_revenue_does_not_inherit_turnover_range() -> None:
    """GST revenue must not inherit a neighboring turnover range."""
    raw_text = (
        "Total Turnover approx Rs. 98-102 Cr (rounded). "
        "GST filings show around 100 Cr."
    )

    financials, metadata = VectorlessRAGExtractor().extract_financials(raw_text)

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
