"""Optional Anthropic web-search provider.

The Anthropic SDK is imported only when a provider is instantiated.
Search results and model-generated findings are treated as untrusted.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlparse


DEFAULT_MODEL = "claude-sonnet-4-20250514"


def _get(value: Any, key: str, default: Any = None) -> Any:
    """Read a field from either an SDK object or a dictionary."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _is_http_url(value: Any) -> bool:
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


def _extract_company(queries: list[str]) -> str:
    """Extract the quoted company name from the generated research query."""
    if not queries:
        return ""

    match = re.match(r'\s*"([^"]+)"', queries[0])
    return match.group(1).strip() if match else ""


def _extract_search_sources(response: Any) -> dict[str, dict[str, Any]]:
    """Collect URLs actually returned by Anthropic's web-search tool."""
    sources: dict[str, dict[str, Any]] = {}

    for block in _get(response, "content", []) or []:
        if _get(block, "type") != "web_search_tool_result":
            continue

        for item in _get(block, "content", []) or []:
            url = _get(item, "url")
            if not _is_http_url(url):
                continue

            url = url.strip()
            sources[url] = {
                "url": url,
                "title": _get(item, "title"),
                "published_at": (
                    _get(item, "page_age")
                    or _get(item, "published_at")
                ),
            }

    return sources


def _extract_text(response: Any) -> str:
    parts = []
    for block in _get(response, "content", []) or []:
        if _get(block, "type") == "text":
            value = _get(block, "text")
            if isinstance(value, str):
                parts.append(value)

    return "\n".join(parts)


def _parse_findings(text: str) -> list[dict[str, Any]]:
    """Parse the JSON findings without trusting their verification claims."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)

    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return []

    findings = payload.get("signals") if isinstance(payload, dict) else None
    if not isinstance(findings, list):
        return []

    return [item for item in findings if isinstance(item, dict)]


class AnthropicWebSearchProvider:
    """Search the web through Anthropic's server-side web-search tool."""

    def __init__(self, client: Any, model: str = DEFAULT_MODEL) -> None:
        self.client = client
        self.model = model

    def search(self, queries: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for query in queries:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1600,
                tools=[
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 5,
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Research this company and query: "
                            f"{query}\n\n"
                            "Use web search. Report only material, "
                            "source-supported findings relevant to this "
                            "company. Do not infer misconduct from an "
                            "unrelated company or an unverified allegation. "
                            "Return only JSON in this schema:\n"
                            '{"signals":[{"type":"legal|fraud|operational|'
                            'market|sector","severity":"low|medium|high|'
                            'critical","confidence":0.0,"evidence":"brief '
                            'factual summary","source_url":"exact URL from '
                            'search results","source_title":"title",'
                            '"published_at":"date if available"}]}\n'
                            "Confidence is a qualitative assessment, not a "
                            "calibrated probability. Do not invent URLs, "
                            "dates, facts, or sources."
                        ),
                    }
                ],
            )

            returned_sources = _extract_search_sources(response)
            findings = _parse_findings(_extract_text(response))
            company = _extract_company([query]).casefold()

            for finding in findings:
                url = finding.get("source_url")
                evidence = finding.get("evidence")
                if not isinstance(url, str):
                    continue

                url = url.strip()
                source = returned_sources.get(url)
                if source is None:
                    # Reject URLs that were not returned by the search tool.
                    continue

                if not isinstance(evidence, str) or not evidence.strip():
                    continue

                # A conservative identity screen, not proof of identity:
                # require the company name in source metadata or evidence.
                searchable_text = " ".join(
                    str(value or "")
                    for value in (
                        source.get("title"),
                        evidence,
                    )
                ).casefold()

                if not company or company not in searchable_text:
                    continue

                results.append(
                    {
                        "type": finding.get("type"),
                        "severity": finding.get("severity"),
                        "confidence": finding.get("confidence"),
                        "evidence": evidence.strip(),
                        "source_url": url,
                        "source_title": (
                            source.get("title")
                            or finding.get("source_title")
                        ),
                        "published_at": (
                            source.get("published_at")
                            or finding.get("published_at")
                        ),
                        # This flag means URL provenance was checked against
                        # the search-tool output; it does not mean the source
                        # or its claims have been independently authenticated.
                        "source_verified": True,
                        "company_match": True,
                    }
                )

        return results


def create_provider_from_env() -> AnthropicWebSearchProvider | None:
    """Create the provider lazily; return None when not configured."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        return None

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL).strip()
    return AnthropicWebSearchProvider(
        client=anthropic.Anthropic(api_key=api_key),
        model=model or DEFAULT_MODEL,
    )
