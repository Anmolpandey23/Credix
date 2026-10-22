from typing import Any

import pytest

from risk_analysis.research_agent import (
    build_research_queries,
    run_company_research,
)


VALID_RESULT = {
    "type": "legal",
    "severity": "high",
    "confidence": 0.82,
    "evidence": "A public filing reports a pending proceeding.",
    "source_url": "https://example.com/company-filing",
    "source_title": "Company filing",
    "published_at": "2026-09-10",
    "company_match": True,
    "source_verified": True,
}


class FakeProvider:
    def __init__(self, results: list[Any] | None = None, error=None):
        self.results = results if results is not None else []
        self.error = error
        self.queries = []

    def search(self, queries):
        self.queries = queries
        if self.error:
            raise self.error
        return self.results


def test_queries_cover_relevant_research_topics():
    queries = build_research_queries("Example Manufacturing Ltd")

    assert len(queries) == 5
    assert all("Example Manufacturing Ltd" in q for q in queries)
    assert any("regulatory" in q for q in queries)


def test_missing_provider_does_not_claim_live_research():
    result = run_company_research("Example Manufacturing Ltd")

    assert result["status"] == "not_configured"
    assert result["web_research_performed"] is False
    assert result["signals"] == []


def test_valid_finding_preserves_source_provenance():
    provider = FakeProvider([VALID_RESULT])

    result = run_company_research("Example Manufacturing Ltd", provider)

    assert result["status"] == "success"
    assert result["web_research_performed"] is True
    assert result["signals"][0]["source"] == "web"
    assert result["findings"][0]["source_url"] == VALID_RESULT["source_url"]
    assert result["sources"][0]["url"] == VALID_RESULT["source_url"]


@pytest.mark.parametrize(
    "change",
    [
        {"company_match": False},
        {"source_verified": False},
        {"source_url": "javascript:alert(1)"},
        {"source_url": "not-a-url"},
        {"evidence": ""},
        {"type": "unknown-category"},
        {"severity": "extreme"},
        {"confidence": 1.5},
        {"confidence": True},
    ],
)
def test_unverified_or_malformed_findings_are_rejected(change):
    provider = FakeProvider([{**VALID_RESULT, **change}])

    result = run_company_research("Example Manufacturing Ltd", provider)

    assert result["status"] == "no_usable_evidence"
    assert result["signals"] == []
    assert result["web_research_performed"] is True


def test_empty_search_is_not_interpreted_as_zero_risk():
    result = run_company_research(
        "Example Manufacturing Ltd",
        FakeProvider([]),
    )

    assert result["status"] == "no_results"
    assert result["web_research_performed"] is True
    assert result["signals"] == []
    assert any("does not establish" in w for w in result["warnings"])


def test_provider_failure_is_reported_honestly():
    result = run_company_research(
        "Example Manufacturing Ltd",
        FakeProvider(error=TimeoutError()),
    )

    assert result["status"] == "failed"
    assert result["web_research_performed"] is True
    assert result["signals"] == []
    assert "TimeoutError" in result["warnings"][0]


def test_invalid_company_name_is_rejected():
    with pytest.raises(ValueError):
        run_company_research("   ")


def test_max_findings_limit_is_enforced():
    provider = FakeProvider([
        {**VALID_RESULT, "source_url": f"https://example.com/{i}"}
        for i in range(4)
    ])

    result = run_company_research(
        "Example Manufacturing Ltd",
        provider,
        max_findings=2,
    )

    assert len(result["findings"]) == 2
