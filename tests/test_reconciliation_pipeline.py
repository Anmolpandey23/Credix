import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ai-brain"))

from main import AIBrainPipeline  # noqa: E402


def _make_pipeline(tmp_path, monkeypatch, documents, candidates):
    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")

    monkeypatch.setattr(
        pipeline.extraction,
        "extract_documents",
        lambda file_paths: documents,
    )
    monkeypatch.setattr(
        pipeline.vectorless_rag,
        "extract_document_candidates",
        lambda extracted_documents: candidates,
    )
    return pipeline


def test_pipeline_reconciliation_preserves_context_and_provenance(
    tmp_path, monkeypatch
):
    documents = [
        {
            "source": "annual_report.pdf",
            "page": 2,
            "text": (
                "Financial Year FY 2025-26. "
                "All amounts are stated in INR crore. "
                "Revenue: 100"
            ),
        },
        {
            "source": "gst_return.pdf",
            "page": 3,
            "text": (
                "Financial Year FY 2025-26. "
                "All amounts are stated in INR crore. "
                "Revenue: 102"
            ),
        },
    ]

    candidates = [
        {
            "field": "revenue",
            "value": 100.0,
            "source": "annual_report.pdf",
            "page": 2,
            "evidence_text": "basis: audited revenue; Revenue: 100",
        },
        {
            "field": "revenue",
            "value": 102.0,
            "source": "gst_return.pdf",
            "page": 3,
            "evidence_text": "basis: audited revenue; Revenue: 102",
        },
    ]

    pipeline = _make_pipeline(
        tmp_path, monkeypatch, documents, candidates
    )
    result = pipeline._build_reconciliation(
        ["annual_report.pdf", "gst_return.pdf"]
    )

    assert result["status"] == "completed"
    assert result["observation_count"] == 2
    assert result["comparison_count"] == 1

    observations = result["observations"]
    assert observations[0]["source"] == "annual_report.pdf"
    assert observations[0]["page"] == 2
    assert observations[0]["period"] == "FY2025-2026"
    assert observations[0]["currency"] == "INR"
    assert observations[0]["unit"] == "crore"
    assert observations[0]["evidence_text"] == candidates[0]["evidence_text"]

    assert observations[1]["source"] == "gst_return.pdf"
    assert observations[1]["page"] == 3
    assert observations[1]["period"] == "FY2025-2026"
    assert observations[1]["currency"] == "INR"
    assert observations[1]["unit"] == "crore"

    comparison = result["comparisons"][0]
    assert comparison["source_a"] == "annual_report.pdf"
    assert comparison["source_b"] == "gst_return.pdf"
    assert comparison["page_a"] == 2
    assert comparison["page_b"] == 3
    assert comparison["status"] == "consistent"


def test_pipeline_does_not_guess_missing_financial_context(
    tmp_path, monkeypatch
):
    documents = [
        {
            "source": "report_a.pdf",
            "page": 1,
            "text": "Revenue: 100",
        },
        {
            "source": "report_b.pdf",
            "page": 4,
            "text": "Revenue: 150",
        },
    ]

    candidates = [
        {
            "field": "revenue",
            "value": 100.0,
            "source": "report_a.pdf",
            "page": 1,
            "evidence_text": "Revenue: 100",
        },
        {
            "field": "revenue",
            "value": 150.0,
            "source": "report_b.pdf",
            "page": 4,
            "evidence_text": "Revenue: 150",
        },
    ]

    pipeline = _make_pipeline(
        tmp_path, monkeypatch, documents, candidates
    )
    result = pipeline._build_reconciliation(
        ["report_a.pdf", "report_b.pdf"]
    )

    assert result["observation_count"] == 2

    for observation in result["observations"]:
        assert observation["period"] is None
        assert observation["currency"] is None
        assert observation["unit"] is None
        assert observation["basis"] is None

    comparison = result["comparisons"][0]
    assert comparison["status"] == "insufficient_evidence"
    assert comparison["difference_ratio"] is None
    assert comparison["source_a"] == "report_a.pdf"
    assert comparison["source_b"] == "report_b.pdf"
    assert comparison["page_a"] == 1
    assert comparison["page_b"] == 4
