from unittest.mock import MagicMock, patch

import pytest
from urllib.error import URLError

from risk_analysis.rss_research_provider import (
    GoogleNewsRSSProvider,
)


SAMPLE_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News</title>
    <item>
      <title>Acme Industries faces fraud investigation</title>
      <link>https://news.example.com/acme-fraud</link>
      <pubDate>Mon, 12 Oct 2026 10:00:00 GMT</pubDate>
      <description>Acme Industries is under a reported fraud investigation.</description>
      <source url="https://news.example.com">Example News</source>
    </item>
    <item>
      <title>Other Company faces fraud investigation</title>
      <link>https://news.example.com/other-fraud</link>
      <pubDate>Mon, 12 Oct 2026 11:00:00 GMT</pubDate>
      <description>Other Company is under a reported fraud investigation.</description>
      <source url="https://news.example.com">Example News</source>
    </item>
    <item>
      <title>Acme Industries announces new office</title>
      <link>https://news.example.com/acme-office</link>
      <pubDate>Mon, 12 Oct 2026 12:00:00 GMT</pubDate>
      <description>Acme Industries opens a new office.</description>
      <source url="https://news.example.com">Example News</source>
    </item>
  </channel>
</rss>
"""


def test_rss_parser_preserves_source_and_date():
    provider = GoogleNewsRSSProvider()

    response = MagicMock()
    response.read.return_value = SAMPLE_RSS
    response.__enter__.return_value = response

    with patch(
        "risk_analysis.rss_research_provider.urllib.request.urlopen",
        return_value=response,
    ):
        items = provider._fetch('"Acme Industries" fraud')

    assert len(items) == 3
    assert items[0]["url"] == "https://news.example.com/acme-fraud"
    assert items[0]["published_at"] == (
        "Mon, 12 Oct 2026 10:00:00 GMT"
    )
    assert items[0]["source_title"] == "Example News"


def test_search_filters_unrelated_company_and_non_risk_news():
    provider = GoogleNewsRSSProvider()

    with patch.object(
        provider,
        "_fetch",
        return_value=[
            {
                "title": "Acme Industries faces fraud investigation",
                "url": "https://news.example.com/acme-fraud",
                "published_at": "2026-10-12",
                "description": "Acme Industries is under a fraud investigation.",
                "source_title": "Example News",
            },
            {
                "title": "Other Company faces fraud investigation",
                "url": "https://news.example.com/other-fraud",
                "published_at": "2026-10-12",
                "description": "Other Company is under a fraud investigation.",
                "source_title": "Example News",
            },
            {
                "title": "Acme Industries opens a new office",
                "url": "https://news.example.com/acme-office",
                "published_at": "2026-10-12",
                "description": "Acme Industries opens a new office.",
                "source_title": "Example News",
            },
        ],
    ):
        findings = provider.search(
            ['"Acme Industries" legal proceedings']
        )

    assert len(findings) == 1
    assert findings[0]["source_url"] == (
        "https://news.example.com/acme-fraud"
    )
    assert findings[0]["type"] == "fraud"
    assert findings[0]["published_at"] == "2026-10-12"
    assert findings[0]["company_match"] is True
    assert findings[0]["source_verified"] is True


def test_repeated_urls_are_deduplicated():
    provider = GoogleNewsRSSProvider()
    article = {
        "title": "Acme Industries faces fraud investigation",
        "url": "https://news.example.com/acme-fraud",
        "published_at": "2026-10-12",
        "description": "Acme Industries is under a fraud investigation.",
        "source_title": "Example News",
    }

    with patch.object(provider, "_fetch", return_value=[article]):
        findings = provider.search(
            ['"Acme Industries" legal proceedings',
             '"Acme Industries" regulatory actions']
        )

    assert len(findings) == 1


def test_network_errors_are_available_for_agent_to_handle():
    provider = GoogleNewsRSSProvider()

    with patch.object(
        provider,
        "_fetch",
        side_effect=URLError("network unavailable"),
    ):
        try:
            provider.search(['"Acme Industries" legal proceedings'])
        except URLError:
            pass
        else:
            raise AssertionError("Expected network error to propagate")


@pytest.mark.parametrize(
    "title,description",
    [
        (
            "Court dismisses fraud charges against Acme Industries",
            "The court dismissed the fraud charges against Acme Industries.",
        ),
        (
            "Court overturns earlier fraud finding against Acme Industries",
            "An earlier fraud finding against Acme Industries was overturned.",
        ),
    ],
)
def test_exculpatory_headlines_are_not_adverse_findings(
    title, description
):
    provider = GoogleNewsRSSProvider()
    article = {
        "title": title,
        "url": "https://news.example.com/acme-court-update",
        "published_at": "2026-10-12",
        "description": description,
        "source_title": "Example News",
    }

    with patch.object(provider, "_fetch", return_value=[article]):
        findings = provider.search(
            ['"Acme Industries" fraud']
        )

    assert findings == []


def test_mixed_outcome_article_preserves_separate_active_investigation():
    provider = GoogleNewsRSSProvider()
    article = {
        "title": (
            "Court dismisses old fraud case; new bribery investigation "
            "continues at Acme Industries"
        ),
        "url": "https://news.example.com/acme-mixed-outcome",
        "published_at": "2026-10-12",
        "description": (
            "The court dismissed an older fraud case. Separately, "
            "authorities continue an active bribery investigation "
            "involving Acme Industries."
        ),
        "source_title": "Example News",
    }

    with patch.object(provider, "_fetch", return_value=[article]):
        findings = provider.search(
            ['"Acme Industries" legal proceedings']
        )

    assert len(findings) == 1
    assert findings[0]["type"] == "legal"
    assert "bribery investigation" in findings[0]["evidence"].casefold()
