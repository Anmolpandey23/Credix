"""Deterministic Credit Appraisal Memo generation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


def _money_value(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "Not available"
    return f"{value:,.2f}"


def _markdown_bullets(items: list[Any], empty_text: str) -> str:
    clean_items = [str(item).strip() for item in items if str(item).strip()]
    if not clean_items:
        return f"- {empty_text}"
    return "\n".join(f"- {item}" for item in clean_items)


def generate_credit_appraisal_memo(
    *,
    company: str,
    financials: Mapping[str, Any],
    ai_brain_assessment: Mapping[str, Any] | None = None,
    intellicredit_assessment: Mapping[str, Any] | None = None,
    reconciliation: Mapping[str, Any] | None = None,
    inspection: Mapping[str, Any] | None = None,
    research: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a traceable draft memo; do not make a lending decision."""
    ai = dict(ai_brain_assessment or {})
    credit = dict(intellicredit_assessment or {})
    reconciliation_data = dict(reconciliation or {})
    inspection_data = dict(inspection or {})
    research_data = dict(research or {})

    breakdown = credit.get("breakdown")
    breakdown = breakdown if isinstance(breakdown, Mapping) else {}

    evidence = credit.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}

    reasons = credit.get("reasons")
    reasons = reasons if isinstance(reasons, list) else []

    warnings = list(credit.get("warnings") or [])
    warnings.extend(inspection_data.get("warnings") or [])

    comparisons = reconciliation_data.get("comparisons")
    comparisons = comparisons if isinstance(comparisons, list) else []
    discrepancies = [
        item for item in comparisons
        if isinstance(item, Mapping) and item.get("status") == "discrepancy"
    ]
    unresolved = [
        item for item in comparisons
        if isinstance(item, Mapping)
        and item.get("status") in {"incomparable", "insufficient_evidence"}
    ]

    # Zero comparisons means consistency has not been established.
    comparison_count = reconciliation_data.get("comparison_count")
    if (
        isinstance(comparison_count, bool)
        or not isinstance(comparison_count, int)
        or comparison_count < 0
    ):
        comparison_count = len(comparisons)

    if comparison_count == 0:
        unresolved_empty_text = (
            "No same-field observation pairs were available to compare; "
            "cross-source consistency has not been established."
        )
    else:
        unresolved_empty_text = (
            "No comparisons requiring additional evidence were reported "
            f"among {comparison_count} comparison(s)."
        )

    findings = inspection_data.get("findings")
    findings = findings if isinstance(findings, list) else []

    signals = research_data.get("signals")
    signals = signals if isinstance(signals, list) else []
    sources = research_data.get("sources")
    sources = sources if isinstance(sources, list) else []
    research_status = research_data.get(
        "status",
        "not_performed",
    )

    financial_rows = [
        f"- **{field.replace('_', ' ').title()}:** {_money_value(value)}"
        for field, value in financials.items()
    ]

    evidence_rows = []
    for field, record in evidence.items():
        if not isinstance(record, Mapping):
            continue
        evidence_rows.append(
            f"- **{field}:** value={record.get('value', 'N/A')}; "
            f"source={record.get('source') or 'unknown'}; "
            f"page={record.get('page') or 'unknown'}; "
            f"evidence status={record.get('evidence_status') or 'unknown'}"
        )

    finding_rows = []
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        finding_rows.append(
            f"- **{finding.get('severity', 'unknown').upper()} — "
            f"{finding.get('category', 'other')}:** "
            f"{finding.get('observation', 'No observation recorded')}"
            + (
                f"\n  - Evidence: {finding['evidence']}"
                if finding.get("evidence")
                else "\n  - Evidence: not supplied"
            )
        )

    signal_rows = []
    for signal in signals:
        if not isinstance(signal, Mapping):
            continue
        signal_rows.append(
            f"- **{signal.get('type', 'signal')}** "
            f"({signal.get('severity', 'unknown')}): "
            f"{signal.get('evidence') or 'No supporting evidence supplied'}"
        )

    source_rows = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue

        url = source.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue

        title = source.get("title") or "Untitled source"
        published_at = source.get("published_at") or "Date not supplied"
        source_rows.append(
            f"- **{title}** — published: {published_at}; URL: {url}"
        )

    reconciliation_rows = []
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            continue
        reconciliation_rows.append(
            f"- **{comparison.get('field', 'unknown field')} — "
            f"{comparison.get('status', 'unknown')}:** "
            f"{comparison.get('reason', 'No explanation supplied')}; "
            f"sources: {comparison.get('source_a', 'unknown')} and "
            f"{comparison.get('source_b', 'unknown')}"
        )

    score = credit.get("final_score")
    score_text = (
        f"{score}/100 (higher means higher risk)"
        if isinstance(score, (int, float)) and not isinstance(score, bool)
        else "Unavailable"
    )
    ai_score = ai.get("final_score")
    ai_score_text = (
        f"{ai_score}/100 (AI Brain score; separate scale)"
        if isinstance(ai_score, (int, float)) and not isinstance(ai_score, bool)
        else "Unavailable"
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    markdown = f"""# Credit Appraisal Memo — Draft

**Company:** {company}
**Generated (UTC):** {generated_at}
**Document status:** Draft for human review — not a lending decision

## 1. Executive summary

- **IntelliCredit risk score:** {score_text}
- **IntelliCredit risk category:** {credit.get('risk_category', 'Unavailable')}
- **AI Brain score:** {ai_score_text}
- **IntelliCredit assessment status:** {credit.get('status', 'not_available')}
- **Research status:** {research_status}

The two scores have different meanings and must not be compared as if they were the same scale. This memo does not approve or reject a credit application.

## 2. Financial information extracted

{chr(10).join(financial_rows) if financial_rows else '- No financial fields supplied.'}

## 3. IntelliCredit module breakdown

- **Financial risk:** {breakdown.get('financial', 'Unavailable')}
- **Fraud-risk indicators:** {breakdown.get('fraud', 'Unavailable')}
- **External risk:** {breakdown.get('external', 'Unavailable')}
- **Operational risk:** {breakdown.get('operational', 'Unavailable')}

## 4. Principal reasons and risk indicators

{_markdown_bullets(reasons, 'No reasons supplied by the assessment engine.')}

## 5. Evidence and provenance

{chr(10).join(evidence_rows) if evidence_rows else '- Field-level source evidence is unavailable.'}

## 6. Cross-source reconciliation

**Reconciliation status:** {reconciliation_data.get('status', 'not_run')}

### Reported discrepancies

{_markdown_bullets([f"{item.get('field')}: {item.get('reason', 'Difference reported')}" for item in discrepancies], 'No confirmed comparable-value discrepancies reported.')}

### Comparisons needing more evidence

{_markdown_bullets([f"{item.get('field')}: {item.get('reason', 'Comparison unresolved')}" for item in unresolved], unresolved_empty_text)}

A discrepancy is a review trigger, not proof of fraud. Incomparable values should not be treated as equivalent or conflicting without adequate context.

## 7. Primary inspection

**Inspection status:** {inspection_data.get('status', 'not_provided')}

{_markdown_bullets(finding_rows, 'No inspection findings supplied.')}

## 8. External research

**Research status:** {research_status}

{_markdown_bullets(signal_rows, 'No external research signals supplied.')}

### Research sources

{chr(10).join(source_rows) if source_rows else '- No source URLs supplied.'}

Research claims should be independently checked against their original sources, publication dates, and company identity before a credit decision.

## 9. Limitations and review actions

{_markdown_bullets(list(dict.fromkeys(warnings)), 'No additional warnings supplied by the assessment engine.')}

- Verify the original financial documents and extracted values.
- Resolve material reconciliation issues and document the accounting period, currency, unit, and basis.
- Verify inspection findings and research sources independently.
- Review the applicant's repayment capacity, proposed facility, collateral, policy requirements, and historical repayment information.
- Do not use this draft alone to approve, decline, price, or disburse credit.

## 10. Reviewer decision

**Decision:** Pending authorized human review

**Reviewer notes:** Not supplied
"""

    return {
        "status": "draft",
        "company": company,
        "generated_at": generated_at,
        "decision": "pending_human_review",
        "research_status": research_status,
        "discrepancy_count": len(discrepancies),
        "unresolved_comparison_count": len(unresolved),
        "inspection_finding_count": len(findings),
        "markdown": markdown,
    }
