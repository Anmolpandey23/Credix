from pathlib import Path

import pypdf

from extraction.service import ExtractionService


def test_pdf_extraction_preserves_page_numbers(tmp_path, monkeypatch):
    pdf_path = tmp_path / "company.pdf"
    pdf_path.write_bytes(b"mock pdf")

    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, _path):
            self.pages = [
                FakePage("Revenue: 100"),
                FakePage(""),
                FakePage("Net profit: 20"),
            ]

    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)

    documents = ExtractionService().extract_documents([pdf_path])

    assert len(documents) == 2
    assert documents[0]["source"] == "company.pdf"
    assert documents[0]["page"] == 1
    assert "Revenue: 100" in documents[0]["text"]
    assert documents[1]["source"] == "company.pdf"
    assert documents[1]["page"] == 3
    assert "Net profit: 20" in documents[1]["text"]


def test_csv_and_json_keep_filename_and_unknown_page(tmp_path):
    csv_path = tmp_path / "financials.csv"
    csv_path.write_text(
        "revenue,profit\n100,20\n",
        encoding="utf-8",
    )

    json_path = tmp_path / "financials.json"
    json_path.write_text(
        '{"revenue": 100, "profit": 20}',
        encoding="utf-8",
    )

    documents = ExtractionService().extract_documents(
        [csv_path, json_path]
    )

    assert len(documents) == 2
    assert documents[0]["source"] == "financials.csv"
    assert documents[0]["page"] is None
    assert documents[0]["text"]
    assert documents[1]["source"] == "financials.json"
    assert documents[1]["page"] is None
    assert documents[1]["text"]


def test_pdf_fallback_does_not_guess_page_number(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"mock pdf")

    def failing_reader(_path):
        raise RuntimeError("Simulated PDF page extraction failure")

    monkeypatch.setattr(pypdf, "PdfReader", failing_reader)

    service = ExtractionService()
    monkeypatch.setattr(
        service,
        "extract",
        lambda _path: "Revenue: 100",
    )

    documents = service.extract_documents([pdf_path])

    assert len(documents) == 1
    assert documents[0]["source"] == "scanned.pdf"
    assert documents[0]["page"] is None
    assert documents[0]["text"] == "Revenue: 100"


def _build_pipeline_with_mock_documents(tmp_path, documents):
    from main import AIBrainPipeline

    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")

    class FakeExtraction:
        def extract_documents(self, _file_paths):
            return documents

    pipeline.extraction = FakeExtraction()
    return pipeline


def test_source_evidence_matches_unique_document_page(tmp_path):
    pipeline = _build_pipeline_with_mock_documents(
        tmp_path,
        [
            {
                "source": "annual_report.pdf",
                "page": 2,
                "text": "Revenue: 1250000 INR",
            }
        ],
    )

    result = pipeline._build_source_evidence(
        [tmp_path / "annual_report.pdf"],
        {"revenue": "Revenue: 1250000 INR"},
    )

    assert result["revenue"]["status"] == "matched"
    assert result["revenue"]["source"] == "annual_report.pdf"
    assert result["revenue"]["page"] == 2


def test_source_evidence_marks_duplicate_matches_ambiguous(tmp_path):
    pipeline = _build_pipeline_with_mock_documents(
        tmp_path,
        [
            {
                "source": "annual_report.pdf",
                "page": 1,
                "text": "Revenue: 1250000 INR",
            },
            {
                "source": "annual_report.pdf",
                "page": 3,
                "text": "Revenue: 1250000 INR",
            },
        ],
    )

    result = pipeline._build_source_evidence(
        [tmp_path / "annual_report.pdf"],
        {"revenue": "Revenue: 1250000 INR"},
    )

    assert result["revenue"]["status"] == "ambiguous"
    assert result["revenue"]["source"] is None
    assert result["revenue"]["page"] is None


def test_source_evidence_rejects_numeric_only_context(tmp_path):
    pipeline = _build_pipeline_with_mock_documents(
        tmp_path,
        [
            {
                "source": "annual_report.pdf",
                "page": 2,
                "text": "Revenue: 1250000 INR",
            }
        ],
    )

    result = pipeline._build_source_evidence(
        [tmp_path / "annual_report.pdf"],
        {"revenue": "1250000"},
    )

    assert result["revenue"]["status"] == "insufficient_context"
    assert result["revenue"]["source"] is None
    assert result["revenue"]["page"] is None
