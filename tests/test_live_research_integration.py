import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ai-brain"))

from main import AIBrainPipeline  # noqa: E402


def _financial_file(path: Path) -> Path:
    path.write_text(
        json.dumps({
            "company": "Research Integration Ltd",
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
    return path


class FakeResearchProvider:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.queries = []

    def search(self, queries):
        self.queries = queries
        if self.error:
            raise self.error
        return self.results


VALID_FINDING = {
    "type": "legal",
    "severity": "high",
    "confidence": 0.82,
    "evidence": "A published filing reports a pending proceeding.",
    "source_url": "https://example.com/filing",
    "source_title": "Public filing",
    "published_at": "2026-09-10",
    "company_match": True,
    "source_verified": True,
}


def test_research_is_off_by_default(tmp_path):
    file_path = _financial_file(tmp_path / "financials.json")
    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")

    result = pipeline.run("Research Integration Ltd", [file_path])

    assert "final_score" in result
    assert result["intellicredit_risk"]["web_research_performed"] is False
    assert result["credit_appraisal_memo"]["research_status"] == "not_performed"


def test_opt_in_research_preserves_source_provenance_and_separate_scores(
    tmp_path,
):
    file_path = _financial_file(tmp_path / "financials.json")
    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    provider = FakeResearchProvider([VALID_FINDING])

    baseline = pipeline.run(
        "Research Integration Ltd",
        [file_path],
        perform_web_research=False,
    )
    researched = pipeline.run(
        "Research Integration Ltd",
        [file_path],
        perform_web_research=True,
        research_provider=provider,
    )

    assert researched["final_score"] == baseline["final_score"]
    assert researched["intellicredit_risk"]["web_research_performed"] is True
    assert researched["research"]["status"] == "success"
    assert researched["research"]["sources"][0]["url"] == VALID_FINDING["source_url"]
    assert researched["credit_appraisal_memo"]["research_status"] == "success"
    assert VALID_FINDING["source_url"] in researched["credit_appraisal_memo"]["markdown"]


def test_provider_failure_does_not_crash_pipeline_or_claim_success(
    tmp_path,
):
    file_path = _financial_file(tmp_path / "financials.json")
    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    provider = FakeResearchProvider(error=TimeoutError())

    result = pipeline.run(
        "Research Integration Ltd",
        [file_path],
        perform_web_research=True,
        research_provider=provider,
    )

    assert "final_score" in result
    assert result["intellicredit_risk"]["status"] == "success"
    assert result["research"]["status"] == "failed"
    assert result["research"]["signals"] == []
    assert result["credit_appraisal_memo"]["research_status"] == "failed"
    assert result["intellicredit_risk"]["web_research_performed"] is False


def test_live_research_is_not_used_when_disabled_even_if_provider_supplied(
    tmp_path,
):
    file_path = _financial_file(tmp_path / "financials.json")
    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    provider = FakeResearchProvider([VALID_FINDING])

    result = pipeline.run(
        "Research Integration Ltd",
        [file_path],
        perform_web_research=False,
        research_provider=provider,
    )

    assert provider.queries == []
    assert result["intellicredit_risk"]["web_research_performed"] is False
    assert result["credit_appraisal_memo"]["research_status"] == "not_performed"


def test_research_findings_affect_intellicredit_external_risk(tmp_path):
    file_path = _financial_file(tmp_path / "financials.json")
    pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
    provider = FakeResearchProvider([VALID_FINDING])

    baseline = pipeline.run(
        "Research Integration Ltd",
        [file_path],
        perform_web_research=False,
    )
    researched = pipeline.run(
        "Research Integration Ltd",
        [file_path],
        perform_web_research=True,
        research_provider=provider,
    )

    baseline_external = baseline["intellicredit_risk"]["breakdown"]["external"]
    researched_external = researched["intellicredit_risk"]["breakdown"]["external"]

    assert researched["research"]["status"] == "success"
    assert researched["research"]["signals"]
    assert researched_external != baseline_external
    assert researched["final_score"] == baseline["final_score"]
