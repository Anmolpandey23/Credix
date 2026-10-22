"""No-API-key company news search using Google News RSS.

RSS results are leads for human review, not independently verified facts.
Uses only the Python standard library.
"""

from __future__ import annotations

import re
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import certifi
from collections.abc import Mapping
from html import unescape
from typing import Any


RSS_URL = "https://news.google.com/rss/search"
USER_AGENT = "AIBrain-CompanyResearch/1.0"

# Conservative keywords: ordinary company mentions do not become risk signals.
RISK_KEYWORDS = {
    "fraud": (
        "fraud", "embezzlement", "bribery", "forgery",
        "money laundering", "financial scam",
    ),
    "legal": (
        "lawsuit", "court case", "charged", "indicted",
        "prosecution", "legal proceedings", "arrested",
    ),
    "operational": (
        "insolvency", "bankruptcy", "default", "recall",
        "shutdown", "license revoked", "production halted",
    ),
    "market": (
        "credit downgrade", "rating downgrade", "debt restructuring",
        "missed payment", "payment default",
    ),
}


def _clean_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", unescape(value)).strip()


# These phrases indicate that an adverse allegation or finding may have
# been dismissed, reversed, or rejected. RSS text is incomplete, so this
# is a conservative filter rather than a legal-outcome determination.
EXCULPATORY_PATTERNS = (
    r"\b(?:charges?|case|lawsuit|proceedings?)\s+(?:were\s+|was\s+)?dismissed\b",
    r"\bdismiss(?:es|ed|al)\s+(?:(?:the|a|an|old|older|earlier|previous|former)\s+)*(?:fraud|bribery|forgery|money laundering)?\s*(?:charges?|case|lawsuit|proceedings?)\b",
    r"\b(?:acquitted|acquittal|cleared of|found not guilty)\b",
    r"\b(?:overturned|quashed|set aside|reversed)\b.{0,100}\b(?:fraud|finding|verdict|conviction|order|charges?)\b",
    r"\b(?:fraud|finding|verdict|conviction|order|charges?)\b.{0,100}\b(?:overturned|quashed|set aside|reversed)\b",
)


def _has_exculpatory_context(text: str) -> bool:
    lowered = text.casefold()
    return any(
        re.search(pattern, lowered)
        for pattern in EXCULPATORY_PATTERNS
    )


def _remove_exculpatory_sentences(text: str) -> str:
    """Remove clearly exculpatory sentences before classifying remaining text."""
    sentences = re.split(r"(?<=[.!?])\s+|;\s+", text)
    retained = [
        sentence for sentence in sentences
        if sentence.strip() and not _has_exculpatory_context(sentence)
    ]
    return " ".join(retained)


def _classify(text: str) -> tuple[str, str] | None:
    # A dismissal in one sentence should not hide a separate active issue.
    remaining = _remove_exculpatory_sentences(text)
    lowered = remaining.casefold()

    # An explicitly active legal investigation should not be classified
    # solely from an unrelated fraud allegation in the same article.
    if (
        "bribery investigation" in lowered
        or "active investigation" in lowered
        or "investigation continues" in lowered
    ):
        return "legal", "medium"

    for signal_type, keywords in RISK_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return signal_type, "medium"

    return None


def _extract_company(queries: list[str]) -> str:
    if not queries:
        return ""

    match = re.match(r'\s*"([^"]+)"', queries[0])
    return match.group(1).strip() if match else ""


class GoogleNewsRSSProvider:
    """Fetch company news results without an API key."""

    def __init__(self, timeout: float = 8.0, max_results_per_query: int = 10):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_results_per_query < 1:
            raise ValueError("max_results_per_query must be positive")

        self.timeout = timeout
        self.max_results_per_query = max_results_per_query

    def _fetch(self, query: str) -> list[Mapping[str, Any]]:
        params = urllib.parse.urlencode(
            {
                "q": query,
                "hl": "en-IN",
                "gl": "IN",
                "ceid": "IN:en",
            }
        )
        request = urllib.request.Request(
            f"{RSS_URL}?{params}",
            headers={"User-Agent": USER_AGENT},
        )

        ssl_context = ssl.create_default_context(
            cafile=certifi.where()
        )
        with urllib.request.urlopen(
            request,
            timeout=self.timeout,
            context=ssl_context,
        ) as response:
            xml_data = response.read(2_000_000)

        root = ET.fromstring(xml_data)
        results = []

        for item in root.findall("./channel/item")[
            : self.max_results_per_query
        ]:
            title = _clean_html(item.findtext("title") or "")
            link = (item.findtext("link") or "").strip()
            published = (item.findtext("pubDate") or "").strip()
            description = _clean_html(
                item.findtext("description") or ""
            )
            source_node = item.find("source")
            source_title = (
                _clean_html("".join(source_node.itertext()))
                if source_node is not None
                else None
            )

            if not title or not link.startswith(("http://", "https://")):
                continue

            results.append(
                {
                    "title": title,
                    "url": link,
                    "published_at": published or None,
                    "description": description,
                    "source_title": source_title,
                }
            )

        return results

    def search(self, queries: list[str]) -> list[Mapping[str, Any]]:
        company = _extract_company(queries)
        if not company:
            return []

        findings: list[Mapping[str, Any]] = []
        seen_urls: set[str] = set()

        for query in queries:
            for article in self._fetch(query):
                url = article["url"]
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title = article["title"]
                description = article["description"]
                searchable = f"{title} {description}"

                # Require an explicit company-name mention. This is only
                # a basic relevance filter, not proof of legal identity.
                if company.casefold() not in searchable.casefold():
                    continue

                classification = _classify(searchable)
                if classification is None:
                    continue

                signal_type, severity = classification
                evidence = description or title

                findings.append(
                    {
                        "type": signal_type,
                        "severity": severity,
                        "confidence": 0.3,
                        "evidence": evidence,
                        "source_url": url,
                        "source_title": (
                            article["source_title"] or title
                        ),
                        "published_at": article["published_at"],
                        "company_match": True,
                        # Means the URL came from the RSS response only.
                        # It does not authenticate the publisher or claim.
                        "source_verified": True,
                    }
                )

        return findings
