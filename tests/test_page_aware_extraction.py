import pytest

from rag.vectorless import VectorlessRAGExtractor


def test_preserves_source_and_page_for_each_document():
    extractor = VectorlessRAGExtractor()

    records = extractor.extract_document_candidates([
        {
            "source": "annual_report.pdf",
            "page": 3,
            "text": "Total revenue: INR 12 lakh",
        },
        {
            "source": "gst_return.pdf",
            "page": 2,
            "text": "Total revenue: INR 12 lakh",
        },
    ])

    revenue_records = [
        record for record in records
        if record["field"] == "revenue"
    ]

    assert len(revenue_records) == 2
    assert revenue_records[0]["source"] == "annual_report.pdf"
    assert revenue_records[0]["page"] == 3
    assert revenue_records[1]["source"] == "gst_return.pdf"
    assert revenue_records[1]["page"] == 2

    assert revenue_records[0]["value"] == revenue_records[1]["value"]
    assert all(record["evidence_text"] for record in revenue_records)


def test_missing_page_is_not_invented():
    extractor = VectorlessRAGExtractor()

    records = extractor.extract_document_candidates([
        {
            "source": "financials.csv",
            "text": "Revenue: 500000",
        }
    ])

    revenue = next(
        record for record in records
        if record["field"] == "revenue"
    )

    assert revenue["source"] == "financials.csv"
    assert revenue["page"] is None


def test_empty_document_text_produces_no_candidates():
    extractor = VectorlessRAGExtractor()

    assert extractor.extract_document_candidates([
        {"source": "empty.pdf", "page": 1, "text": "   "}
    ]) == []


@pytest.mark.parametrize(
    "document",
    [
        {"source": "", "text": "Revenue: 100"},
        {"source": "report.pdf", "text": 123},
        {"source": "report.pdf", "page": 0, "text": "Revenue: 100"},
        {"source": "report.pdf", "page": True, "text": "Revenue: 100"},
        {"source": "report.pdf", "page": 1.5, "text": "Revenue: 100"},
    ],
)
def test_invalid_document_metadata_is_rejected(document):
    extractor = VectorlessRAGExtractor()

    with pytest.raises(ValueError):
        extractor.extract_document_candidates([document])


def test_existing_extraction_api_remains_available():
    extractor = VectorlessRAGExtractor()

    financials, metadata = extractor.extract_financials(
        "Total revenue: INR 12 lakh"
    )

    assert financials["revenue"] is not None
    assert metadata["revenue"] is not None
