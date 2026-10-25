"""Provider-independent company research with evidence validation.

This module does not perform live searches by itself. A search provider
must be explicitly supplied. Provider output is untrusted until validated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeGuard
from urllib.parse import urlparse


ALLOWED_TYPES = {
    "legal",
    "fraud",
    "operational",
    "market",
    "sector",
}

ALLOWED_SEVERITIES = {"low", "medium", "high", "critical"}

RESEARCH_TOPICS = (
    "legal proceedings and court cases",
    "regulatory actions and compliance",
    "promoter and director public records",
    "reputable business news and material events",
    "industry and sector risks",
)


class ResearchProvider(Protocol):
    """Interface implemented by a real or mocked search provider."""

    def search(
        self, queries: list[str]
    ) -> list[Mapping[str, Any]]:
        """Return untrusted search-result records."""
        ...


def _valid_http_url(value: Any) -> TypeGuard[str]:
    if not isinstance(value, str):
        return False

    try:
        parsed = urlparse(value.strip())
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.netloc)
            and not any(char.isspace() for char in value.strip())
        )
    except ValueError:
        return False


def build_research_queries(company_name: str) -> list[str]:
    """Build focused queries without inventing company identifiers."""
    company = company_name.strip()
    if not company:
        raise ValueError("company_name must not be empty")

    return [
        f'"{company}" {topic}'
        for topic in RESEARCH_TOPICS
    ]


def _normalize_finding(
    raw: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Accept only findings with explicit identity and source checks."""
    source_url = raw.get("source_url")
    evidence = raw.get("evidence")
    company_match = raw.get("company_match") is True
    source_verified = raw.get("source_verified") is True

    if not isinstance(source_url, str) or not _valid_http_url(source_url):
        return None

    if not isinstance(evidence, str) or not evidence.strip():
        return None

    if not company_match or not source_verified:
        return None

    signal_type = raw.get("type")
    severity = raw.get("severity")
    confidence = raw.get("confidence")

    if not isinstance(signal_type, str):
        return None

    signal_type = signal_type.strip().lower()
    if signal_type not in ALLOWED_TYPES:
        return None

    if not isinstance(severity, str):
        return None

    severity = severity.strip().lower()
    if severity not in ALLOWED_SEVERITIES:
        return None

    # Reject booleans, NaN, infinity and values outside [0, 1].
    if isinstance(confidence, bool) or not isinstance(
        confidence, (int, float)
    ):
        return None

    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        return None

    title = raw.get("source_title")
    published_at = raw.get("published_at")

    return {
        "type": signal_type,
        "severity": severity,
        "confidence": confidence,
        "source": "web",
        "evidence": evidence.strip(),
        "source_url": source_url.strip(),
        "source_title": (
            title.strip() if isinstance(title, str) and title.strip()
            else None
        ),
        "published_at": (
            published_at.strip()
            if isinstance(published_at, str) and published_at.strip()
            else None
        ),
        "company_match": True,
        "source_verified": True,
    }


def run_company_research(
    company_name: str,
    provider: ResearchProvider | None = None,
    *,
    max_findings: int = 20,
) -> dict[str, Any]:
    """Run research through an explicitly supplied provider.

    Statuses:
      - not_configured: no provider was supplied
      - success: usable evidence was returned
      - no_results: provider returned no results
      - no_usable_evidence: results existed but none passed validation
      - failed: the provider raised an exception
    """
    if not isinstance(company_name, str) or not company_name.strip():
        raise ValueError("company_name must not be empty")

    if isinstance(max_findings, bool) or not isinstance(max_findings, int):
        raise ValueError("max_findings must be a positive integer")

    if max_findings < 1:
        raise ValueError("max_findings must be a positive integer")

    if provider is None:
        return {
            "status": "not_configured",
            "company": company_name.strip(),
            "web_research_performed": False,
            "findings": [],
            "signals": [],
            "sources": [],
            "warnings": [
                "No live research provider is configured. "
                "No conclusion can be drawn from the absence of findings."
            ],
        }

    try:
        raw_results = provider.search(
            build_research_queries(company_name)
        )
        if not isinstance(raw_results, list):
            raise TypeError("Provider search() must return a list")
    except Exception as exc:
        return {
            "status": "failed",
            "company": company_name.strip(),
            "web_research_performed": True,
            "findings": [],
            "signals": [],
            "sources": [],
            "warnings": [
                f"Research provider failed: {type(exc).__name__}. "
                "No conclusion can be drawn from this failure."
            ],
        }

    if not raw_results:
        return {
            "status": "no_results",
            "company": company_name.strip(),
            "web_research_performed": True,
            "findings": [],
            "signals": [],
            "sources": [],
            "warnings": [
                "The provider returned no results. This does not establish "
                "that the company has no adverse events or risks."
            ],
        }

    findings = []
    for raw in raw_results:
        if not isinstance(raw, Mapping):
            continue

        finding = _normalize_finding(raw)
        if finding is not None:
            findings.append(finding)

        if len(findings) >= max_findings:
            break

    sources_by_url = {
        finding["source_url"]: {
            "url": finding["source_url"],
            "title": finding["source_title"],
            "published_at": finding["published_at"],
        }
        for finding in findings
    }

    if not findings:
        return {
            "status": "no_usable_evidence",
            "company": company_name.strip(),
            "web_research_performed": True,
            "findings": [],
            "signals": [],
            "sources": [],
            "warnings": [
                "Search results were returned, but none passed evidence, "
                "source URL, verification and company-identity checks."
            ],
        }

    warnings = [
        "Source and company-match verification flags are provider-supplied "
        "and must be backed by a trusted verification process.",
        "Confidence values are provider assessments, not calibrated "
        "probabilities of default, fraud or misconduct.",
        "Findings require human review before consequential credit decisions.",
    ]

    return {
        "status": "success",
        "company": company_name.strip(),
        "web_research_performed": True,
        "findings": findings,
        "signals": [
            {
                "type": item["type"],
                "severity": item["severity"],
                "confidence": item["confidence"],
                "source": "web",
                "evidence": item["evidence"],
            }
            for item in findings
        ],
        "sources": list(sources_by_url.values()),
        "warnings": warnings,
    }
