
"""Offline adapter connecting AI Brain financial data to IntelliCredit."""

from __future__ import annotations

from typing import Any, Mapping

from .intellicredit_engine.aggregator import (
    aggregate_scores,
    build_score_audit,
    classify_risk,
    compute_missing_penalty,
)
from .intellicredit_engine.external import compute_external_risk
from .intellicredit_engine.explainability import generate_ranked_reasons
from .intellicredit_engine.financial import compute_financial_risk
from .intellicredit_engine.fraud import compute_fraud_risk
from .intellicredit_engine.operational import compute_operational_risk
from .intellicredit_engine.sector import (
    apply_sector_amplification,
    compute_sector_score,
)
from .intellicredit_engine.validator import validate_and_standardize


EVIDENCE_FIELDS = (
    "revenue",
    "total_debt",
    "receivables",
    "gst_revenue",
    "bank_deposits",
    "profit",
    "cash_flow",
)


def _build_financial_evidence(
    mapped_financials: Mapping[str, Any],
    field_metadata: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Build provenance records without changing scoring inputs.

    Expected metadata structure:
        {
            "source_evidence": {
                "revenue": {
                    "source": "...",
                    "page": 1,
                    "evidence_text": "...",
                    "status": "matched",
                }
            },
            "field_details": {
                "revenue": {
                    "confidence_bin": "low",
                    "is_approximate": True,
                    "has_range": True,
                    "min": 98,
                    "max": 102,
                }
            },
        }

    Missing metadata remains unknown. This function does not infer
    reporting period, currency, unit, accounting basis, or comparability.
    """
    metadata = field_metadata or {}
    source_evidence = metadata.get("source_evidence") or {}
    field_details = metadata.get("field_details") or {}

    if not isinstance(source_evidence, Mapping):
        source_evidence = {}

    if not isinstance(field_details, Mapping):
        field_details = {}

    evidence: dict[str, dict[str, Any]] = {}

    for field in EVIDENCE_FIELDS:
        value = mapped_financials.get(field)

        if value is None:
            continue

        # Support both AI Brain and IntelliCredit field names.
        metadata_field = {
            "profit": "net_profit",
            "cash_flow": "operating_cash_flow",
        }.get(field, field)

        source_record = source_evidence.get(metadata_field)
        if not isinstance(source_record, Mapping):
            source_record = source_evidence.get(field)

        if not isinstance(source_record, Mapping):
            source_record = {}

        detail_record = field_details.get(metadata_field)
        if not isinstance(detail_record, Mapping):
            detail_record = field_details.get(field)

        if not isinstance(detail_record, Mapping):
            detail_record = {}

        evidence[field] = {
            "value": value,
            "source": source_record.get("source"),
            "page": source_record.get("page"),
            "evidence_text": source_record.get("evidence_text"),
            "evidence_status": source_record.get("status"),
            "extraction_confidence": detail_record.get("confidence_bin"),
            "is_approximate": detail_record.get("is_approximate"),
            "has_range": detail_record.get("has_range"),
            "min": detail_record.get("min"),
            "max": detail_record.get("max"),
        }

    return evidence


def build_risk_input(
    company_id: str,
    financials: Mapping[str, Any],
    operational_data: Mapping[str, Any] | None = None,
    external_signals: list[dict[str, Any]] | None = None,
    field_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map AI Brain fields to IntelliCredit without inventing values.

    Field metadata is retained separately for auditability. It is not
    passed into the financial scoring model.
    """
    source = dict(financials)

    mapped_financials = {
        # Matching fields.
        "gst_revenue": source.get("gst_revenue"),
        "bank_deposits": source.get("bank_deposits"),

        # Explicit mappings from AI Brain to IntelliCredit.
        "profit": source.get("net_profit", source.get("profit")),
        "cash_flow": source.get(
            "operating_cash_flow",
            source.get("cash_flow"),
        ),

        # Optional fields are accepted only when supplied.
        "net_operating_income": source.get("net_operating_income"),
        "debt_payments": source.get("debt_payments"),
        "current_assets": source.get("current_assets"),
        "current_liabilities": source.get("current_liabilities"),
        "EBIT": source.get("EBIT"),
        "interest_expense": source.get("interest_expense"),
        "revenue_history": source.get("revenue_history"),
        "revenue": source.get("revenue"),
        "total_debt": source.get("total_debt"),
        "receivables": source.get("receivables"),

        # These flags must be explicitly supplied and verified.
        # Evidence presence alone must not enable a ratio.
        "debt_revenue_comparable": source.get(
            "debt_revenue_comparable"
        ),
        "receivables_revenue_comparable": source.get(
            "receivables_revenue_comparable"
        ),
        "gst_bank_comparable": source.get("gst_bank_comparable"),
        "profit_cashflow_comparable": source.get(
            "profit_cashflow_comparable"
        ),
        "revenue_history_comparable": source.get(
            "revenue_history_comparable"
        ),
        "expense_comparable": source.get("expense_comparable"),
    }

    return {
        "company_id": company_id,
        "financials": mapped_financials,
        "operational_data": dict(operational_data or {}),
        "external_signals": list(external_signals or []),

        # Audit metadata remains separate from scoring inputs.
        "field_metadata": dict(field_metadata or {}),
    }


def analyze_offline_risk(
    company_id: str,
    financials: Mapping[str, Any],
    operational_data: Mapping[str, Any] | None = None,
    external_signals: list[dict[str, Any]] | None = None,
    field_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deterministic IntelliCredit analysis without web research.

    A higher IntelliCredit score means higher risk. This score is
    independent of AI Brain's financial-health score.

    Field metadata enriches output provenance but does not alter
    financial values, comparability flags, or scores.
    """
    raw_input = build_risk_input(
        company_id=company_id,
        financials=financials,
        operational_data=operational_data,
        external_signals=external_signals,
        field_metadata=field_metadata,
    )

    validation = validate_and_standardize(raw_input)

    if validation.status != "ok" or validation.data is None:
        return {
            "status": "failed",
            "engine": "IntelliCredit",
            "reason": validation.message or "Risk input validation failed",
            "warnings": list(validation.warnings),
            "web_research_performed": False,
        }

    risk_input = validation.data
    warnings = list(validation.warnings)

    missing_penalty, missing_reasons = compute_missing_penalty(risk_input)

    financial_score, financial_reasons = compute_financial_risk(
        risk_input.financials
    )

    fraud_score, fraud_reasons = compute_fraud_risk(
        risk_input.financials
    )

    external_score, external_reasons = compute_external_risk(
        risk_input.external_signals
    )

    operational_score, operational_reasons = compute_operational_risk(
        risk_input.operational_data
    )

    # Only amplify financial risk when an explicit sector signal exists.
    has_sector_signal = any(
        signal.type == "sector"
        for signal in risk_input.external_signals
    )

    if has_sector_signal:
        sector_score = compute_sector_score(risk_input.external_signals)
        financial_score = apply_sector_amplification(
            financial_score,
            sector_score,
        )

    # Preserve the existing production aggregation formula.
    final_score = aggregate_scores(
        financial_score,
        fraud_score,
        external_score,
        operational_score,
        missing_penalty,
    )

    # Independently expose the arithmetic used to reproduce this score.
    score_audit = build_score_audit(
        financial_score,
        fraud_score,
        external_score,
        operational_score,
        missing_penalty,
    )

    # Fail safely rather than returning a misleading audit.
    if round(float(final_score), 2) != score_audit[
        "final_score_after_clamping"
    ]:
        raise RuntimeError(
            "Score audit does not match the aggregate score"
        )

    reasons = generate_ranked_reasons(
        financial_reasons,
        fraud_reasons,
        external_reasons,
        operational_reasons,
        missing_reasons,
    )

    if not risk_input.external_signals:
        warnings.append(
            "No external research signals were supplied; the external "
            "score is a neutral placeholder, not a verified finding."
        )

    if missing_reasons:
        warnings.append(
            "Some expected financial or operational data is unavailable. "
            "Missing data is a limitation, not proof of fraud or concealment."
        )

    mapped_financials = raw_input["financials"]

    # Non-None mapped fields are not necessarily fields that directly
    # contributed to a ratio or changed the final score.
    fields_used = [
        name
        for name, value in mapped_financials.items()
        if value is not None
    ]

    financial_evidence = _build_financial_evidence(
        mapped_financials=mapped_financials,
        field_metadata=raw_input["field_metadata"],
    )

    if not raw_input["field_metadata"]:
        warnings.append(
            "Financial extraction provenance was not supplied; "
            "source-level evidence could not be attached."
        )

    return {
        "status": "success",
        "engine": "IntelliCredit",
        "score_direction": "higher_score_means_higher_risk",
        "final_score": round(float(final_score), 2),
        "score_audit": score_audit,
        "risk_category": classify_risk(final_score),
        "breakdown": {
            "financial": round(float(financial_score), 2),
            "fraud": round(float(fraud_score), 2),
            "external": round(float(external_score), 2),
            "operational": round(float(operational_score), 2),
        },
        "missing_data_penalty": float(missing_penalty),
        "reasons": reasons,
        "warnings": list(dict.fromkeys(warnings)),
        "financial_fields_used": fields_used,
        "evidence": financial_evidence,
        "web_research_performed": False,
    }
