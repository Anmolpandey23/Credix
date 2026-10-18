import numpy as np
import pytest

from rag.semantic import SemanticVectorRAG


class FakeEmbeddingModel:
    """Deterministic embeddings for isolated unit tests."""

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        vectors = []

        for text in texts:
            lower = text.lower()

            if "revenue" in lower or "sales" in lower:
                vector = [1.0, 0.0, 0.0]
            elif "debt" in lower or "borrowing" in lower:
                vector = [0.0, 1.0, 0.0]
            else:
                vector = [0.0, 0.0, 1.0]

            vectors.append(vector)

        result = np.asarray(vectors, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.maximum(norms, 1e-12)

        return result


@pytest.fixture
def retriever():
    return SemanticVectorRAG(model=FakeEmbeddingModel())


def test_empty_documents_create_empty_index(retriever):
    assert retriever.add_documents([]) == 0
    assert retriever.search("revenue") == []


def test_search_preserves_source_and_page(retriever):
    count = retriever.add_documents([
        {
            "source": "annual_report.pdf",
            "page": 4,
            "text": "Revenue increased during the financial year.",
        },
        {
            "source": "bank_statement.pdf",
            "page": 2,
            "text": "Outstanding debt and borrowing obligations.",
        },
    ])

    assert count == 2

    results = retriever.search("sales", top_k=1)

    assert len(results) == 1
    assert results[0].source == "annual_report.pdf"
    assert results[0].page == 4
    assert results[0].score == pytest.approx(1.0)


def test_unknown_page_is_preserved_as_none(retriever):
    retriever.add_documents([
        {
            "source": "financials.csv",
            "page": None,
            "text": "Revenue figures from the ledger.",
        }
    ])

    result = retriever.search("sales", top_k=1)[0]

    assert result.source == "financials.csv"
    assert result.page is None


def test_blank_query_returns_no_results(retriever):
    retriever.add_documents([
        {"source": "report.pdf", "page": 1, "text": "Revenue increased."}
    ])

    assert retriever.search("   ") == []


def test_top_k_and_min_score_are_respected(retriever):
    retriever.add_documents([
        {"source": "sales.csv", "page": None, "text": "Revenue increased."},
        {"source": "debt.csv", "page": None, "text": "Debt increased."},
    ])

    results = retriever.search("sales", top_k=1, min_score=0.5)

    assert len(results) == 1
    assert results[0].source == "sales.csv"

    with pytest.raises(ValueError):
        retriever.search("sales", min_score=1.1)


def test_invalid_chunk_settings_are_rejected():
    with pytest.raises(ValueError):
        SemanticVectorRAG(chunk_size=10, chunk_overlap=10)

    with pytest.raises(ValueError):
        SemanticVectorRAG(chunk_size=0, chunk_overlap=0)


def test_reindex_replaces_previous_documents(retriever):
    retriever.add_documents([
        {"source": "old.pdf", "page": 1, "text": "Revenue increased."}
    ])

    retriever.add_documents([
        {"source": "new.pdf", "page": 3, "text": "Debt increased."}
    ])

    results = retriever.search("debt", top_k=5)

    assert len(results) == 1
    assert results[0].source == "new.pdf"


class SelectiveFailureModel:
    """Fail indexing selected documents while allowing queries to work."""

    def __init__(self):
        self.delegate = FakeEmbeddingModel()

    def encode(self, texts, **kwargs):
        if any(text.startswith("BROKEN_INDEX:") for text in texts):
            raise RuntimeError("Simulated embedding failure")
        return self.delegate.encode(texts, **kwargs)


def test_failed_reindex_preserves_previous_valid_index():
    model = SelectiveFailureModel()
    retriever = SemanticVectorRAG(model=model)

    retriever.add_documents([
        {
            "source": "original.pdf",
            "page": 5,
            "text": "Revenue increased during the year.",
        }
    ])

    with pytest.raises(RuntimeError, match="Simulated embedding failure"):
        retriever.add_documents([
            {
                "source": "replacement.pdf",
                "page": 9,
                "text": "BROKEN_INDEX: replacement document",
            }
        ])

    results = retriever.search("sales", top_k=5)

    assert len(results) == 1
    assert results[0].source == "original.pdf"
    assert results[0].page == 5


class ZeroVectorModel:
    """Simulate a broken model that returns zero-length embeddings."""

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        return np.zeros((len(texts), 3), dtype=np.float32)


def test_zero_length_document_embeddings_are_rejected():
    retriever = SemanticVectorRAG(model=ZeroVectorModel())

    with pytest.raises(ValueError, match="zero-length vector"):
        retriever.add_documents([
            {
                "source": "report.pdf",
                "page": 1,
                "text": "Some financial evidence.",
            }
        ])

    assert retriever.search("financial evidence") == []


def test_invalid_page_numbers_are_normalized_to_none(retriever):
    retriever.add_documents([
        {
            "source": "report.pdf",
            "page": 0,
            "text": "Revenue increased during the year.",
        }
    ])

    result = retriever.search("sales", top_k=1)[0]

    assert result.page is None


def test_turnover_label_value_and_qualification_stay_together():
    retriever = SemanticVectorRAG(model=FakeEmbeddingModel())
    financial_line = (
        "Total Turnover approx Rs. 98-102 Cr (rounded). "
        "GST filings show around 100 Cr."
    )

    chunks = retriever._split_text(financial_line)

    assert chunks == [financial_line]
    assert "98-102 Cr" in chunks[0]
    assert "approx" in chunks[0]
    assert "(rounded)" in chunks[0]


def test_separate_financial_statement_lines_remain_separate():
    retriever = SemanticVectorRAG(model=FakeEmbeddingModel())
    text = (
        "Total borrowings touching 85-95 Cr range.\n"
        "Cash Flow from Ops is weak ~2-3 Cr."
    )

    chunks = retriever._split_text(text)

    assert chunks == [
        "Total borrowings touching 85-95 Cr range.",
        "Cash Flow from Ops is weak ~2-3 Cr.",
    ]


def test_financial_heading_is_paired_with_numeric_value_and_qualification():
    retriever = SemanticVectorRAG(model=FakeEmbeddingModel())
    text = (
        "Cash Flow from Operations:\n"
        "~2 Cr to 3 Cr only (low due to WC blockage)\n"
        "Total Borrowings / Debt:\n"
        "~85 - 95 Cr (incl. term loans + CC limits)\n"
        "Trade Receivables:\n"
        "65 - 75 Cr (collections delayed)"
    )

    assert retriever._split_text(text) == [
        "Cash Flow from Operations: ~2 Cr to 3 Cr only "
        "(low due to WC blockage)",
        "Total Borrowings / Debt: ~85 - 95 Cr "
        "(incl. term loans + CC limits)",
        "Trade Receivables: 65 - 75 Cr (collections delayed)",
    ]


def test_metric_value_table_becomes_distinct_financial_evidence():
    retriever = SemanticVectorRAG(model=FakeEmbeddingModel())
    text = (
        "Financial Statements\n"
        "Metric\n"
        "Value (Crore)\n"
        "Revenue\n100\n"
        "GST Declared Revenue\n100\n"
        "Bank Deposits\n62\n"
        "Operating Cash Flow\n3\n"
        "Total Debt\n90\n"
        "Notes & Observations\n"
        "- High dependency on credit sales."
    )

    chunks = retriever._split_text(text)

    assert "Revenue: 100 Crore" in chunks
    assert "GST Declared Revenue: 100 Crore" in chunks
    assert "Bank Deposits: 62 Crore" in chunks
    assert "Operating Cash Flow: 3 Crore" in chunks
    assert "Total Debt: 90 Crore" in chunks
    assert "- High dependency on credit sales." in chunks

    assert "Revenue: 100 Crore" != "GST Declared Revenue: 100 Crore"
    assert not any(
        "Notes & Observations" in chunk and "Revenue: 100 Crore" in chunk
        for chunk in chunks
    )


def test_unlabelled_numeric_line_is_not_merged_with_previous_metric():
    retriever = SemanticVectorRAG(model=FakeEmbeddingModel())
    text = (
        "Revenue (approx.) : INR 98 - 102 Cr\n"
        "GST Turnover ~ 100 Cr\n"
        "Bank Credits observed (FY): around 60 to 65 Cr"
    )

    assert retriever._split_text(text) == [
        "Revenue (approx.) : INR 98 - 102 Cr",
        "GST Turnover ~ 100 Cr",
        "Bank Credits observed (FY): around 60 to 65 Cr",
    ]


def test_direct_revenue_query_ranks_revenue_evidence_above_commentary():
    retriever = SemanticVectorRAG()
    retriever._model = None

    documents = [
        {
            "source": "sample.pdf",
            "page": 1,
            "text": (
                "Revenue\n100\n"
                "Notes & Observations\n"
                "- The company reported stable revenue growth over the last year."
            ),
        }
    ]

    retriever.add_documents(documents)

    results = retriever.search("Revenue", top_k=3)

    assert results
    assert results[0].text == "Revenue: 100"


def test_gst_revenue_query_keeps_gst_metric_distinct():
    retriever = SemanticVectorRAG()

    documents = [
        {
            "source": "sample.pdf",
            "page": 1,
            "text": (
                "Metric\nValue (Crore)\n"
                "Revenue\n100\n"
                "GST Declared Revenue\n100\n"
                "Bank Deposits\n62"
            ),
        }
    ]

    retriever.add_documents(documents)

    results = retriever.search("GST Declared Revenue", top_k=3)

    assert results
    assert results[0].text == "GST Declared Revenue: 100 Crore"


def test_debt_query_does_not_replace_debt_evidence_with_other_metrics():
    retriever = SemanticVectorRAG()

    documents = [
        {
            "source": "sample.pdf",
            "page": 1,
            "text": (
                "Metric\nValue (Crore)\n"
                "Revenue\n100\n"
                "Net Profit\n10\n"
                "Total Debt\n90"
            ),
        }
    ]

    retriever.add_documents(documents)

    results = retriever.search("Total Debt", top_k=3)

    assert results
    assert results[0].text == "Total Debt: 90 Crore"


def test_revenue_query_finds_turnover_evidence_in_messy_financial_report():
    retriever = SemanticVectorRAG()

    documents = [
        {
            "source": "messy_company_financials.pdf",
            "page": 1,
            "text": (
                "XYZ INDUSTRIES LTD (Unaudited Report 2024)\n"
                "Total Turnover approx Rs. 98-102 Cr (rounded). "
                "GST filings show around 100 Cr.\n"
                "Net Profit (PAT): near about 9.5 to 10 Cr.\n"
                "Cash Flow from Ops is weak ~2-3 Cr."
            ),
        }
    ]

    retriever.add_documents(documents)
    results = retriever.search("Revenue", top_k=3)

    assert results
    assert "Turnover" in results[0].text


def test_revenue_search_prioritizes_reported_revenue_over_gst_and_commentary(
    retriever,
):
    retriever.add_documents([
        {
            "source": "financials.pdf",
            "page": 1,
            "text": "GST Declared Revenue: 100 Crore",
        },
        {
            "source": "financials.pdf",
            "page": 1,
            "text": "The company reported stable revenue growth over the last year.",
        },
        {
            "source": "financials.pdf",
            "page": 1,
            "text": "Revenue: 100 Crore",
        },
    ])

    results = retriever.search("revenue", top_k=3)

    assert results[0].text == "Revenue: 100 Crore"


def test_gst_revenue_search_prioritizes_gst_declared_revenue(retriever):
    retriever.add_documents([
        {
            "source": "financials.pdf",
            "page": 1,
            "text": "Revenue: 100 Crore",
        },
        {
            "source": "financials.pdf",
            "page": 1,
            "text": "GST Declared Revenue: 100 Crore",
        },
        {
            "source": "financials.pdf",
            "page": 1,
            "text": "The company reported stable revenue growth over the last year.",
        },
    ])

    results = retriever.search("gst revenue", top_k=3)

    assert results[0].text == "GST Declared Revenue: 100 Crore"
