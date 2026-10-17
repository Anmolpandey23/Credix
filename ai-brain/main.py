"""AI Brain pipeline entrypoint for IntelliCredit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal

from config.settings import FINANCIAL_FIELDS, DEFAULT_STORAGE_DIR
from cross_check.rule_engine import RuleEngine
from extraction.service import ExtractionService
from ingestion.service import IngestionService
from insights import FinancialInsightGenerator
from rag.vectorless import VectorlessRAGExtractor
from utils.text_utils import normalize_whitespace


class ConfidenceAnalyzer:
    """Calculates realistic confidence scores based on extraction quality."""

    # Penalty constants
    PENALTY_APPROXIMATE = 0.10  # ~, around, approx
    PENALTY_RANGE = 0.15        # 60-65
    PENALTY_NOISY_TEXT = 0.10   # OCR, fragmented content
    PENALTY_INCOMPLETE = 0.05   # Missing fields
    PENALTY_INCONSISTENCY = 0.10  # Revenue mismatches

    CRITICAL_FIELDS = {"gst_revenue", "bank_deposits"}

    def __init__(self):
        self.field_confidences: Dict[str, Dict[str, Any]] = {}

    def calculate(
        self,
        financials: Dict[str, float | None],
        metadata: Dict,
        flags: List[str],
        extraction_quality: Literal["high", "medium", "low"] = "high"
    ) -> Dict[str, Any]:
        """
        Calculate overall confidence score and per-field metrics.

        Returns:
            {
                "overall_confidence": 0.75,  # 0.0-1.0
                "confidence_level": "medium",  # high (>0.8), medium (0.5-0.8), low (<0.5)
                "field_confidences": {
                    "field_name": {
                        "confidence": 0.85,
                        "extraction_type": "exact|approximate|range",
                        "level": "high|medium|low"
                    }
                },
                "quality_artifacts": {
                    "approx_fields": ["bank_deposits"],
                    "range_fields": ["revenue"],
                    "missing_fields": ["operating_cash_flow"],
                    "critical_approx": True,  # Critical field extracted as approximate
                    "high_approximate_ratio": True  # >50% approximate
                }
            }
        """

        # Step 1: Calculate base score
        extracted_count = sum(1 for v in financials.values() if v is not None)
        base_score = extracted_count / len(FINANCIAL_FIELDS)  # 0.0-1.0

        # Step 2: Analyze each field's extraction type
        approx_fields = []
        range_fields = []
        missing_fields = []

        for field in FINANCIAL_FIELDS:
            if financials.get(field) is None:
                missing_fields.append(field)
            else:
                field_meta = metadata.get(field)
                if field_meta:
                    if field_meta.has_range:
                        range_fields.append(field)
                    elif field_meta.is_approximate:
                        approx_fields.append(field)

        # Step 3: Check for indicator of noisy extraction (quality parameter)
        # If multiple fallback patterns or OCR was used, text quality is lower
        noisy_penalty = 0.10 if extraction_quality == "low" else (0.05 if extraction_quality == "medium" else 0.0)

        # Step 4: Calculate penalties
        penalties = 0.0

        # Approximate value penalty
        if approx_fields:
            approx_ratio = len(approx_fields) / extracted_count if extracted_count > 0 else 0
            penalties += self.PENALTY_APPROXIMATE * approx_ratio

        # Range extraction penalty
        if range_fields:
            range_ratio = len(range_fields) / extracted_count if extracted_count > 0 else 0
            penalties += self.PENALTY_RANGE * range_ratio

        # Incomplete data penalty
        if missing_fields:
            missing_ratio = len(missing_fields) / len(FINANCIAL_FIELDS)
            penalties += self.PENALTY_INCOMPLETE * missing_ratio

        # Noisy text penalty
        penalties += noisy_penalty

        # Inconsistency penalty (from flags)
        inconsistency_flags = {
            "Revenue mismatch - possible inflation",
            "Revenue inconsistency between GST and financials",
        }
        flag_penalty = self.PENALTY_INCONSISTENCY * sum(
            1 for flag in flags if flag in inconsistency_flags
        )
        penalties += flag_penalty

        # Step 5: Check critical field extraction type
        critical_approx = False
        critical_range = False
        for field in self.CRITICAL_FIELDS:
            if field in approx_fields:
                critical_approx = True
                penalties += 0.05  # Extra penalty if critical field is approximate
            elif field in range_fields:
                critical_range = True
                penalties += 0.03  # Extra penalty if critical field is range

        # Step 6: Check if >50% fields are approximate
        high_approx_ratio = False
        if approx_fields and extracted_count > 0:
            approx_pct = len(approx_fields) / extracted_count
            if approx_pct > 0.5:
                high_approx_ratio = True
                penalties += 0.10  # Extra penalty for high approximate ratio

        # Step 7: Calculate final confidence
        overall_confidence = max(0.0, min(1.0, base_score - penalties))

        # Step 8: Determine confidence level
        if overall_confidence >= 0.8:
            confidence_level = "high"
        elif overall_confidence >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # Step 9: Calculate per-field confidence
        field_confidences = {}
        for field in FINANCIAL_FIELDS:
            if financials.get(field) is not None:
                field_meta = metadata.get(field)

                # Determine extraction type
                if field_meta and field_meta.has_range:
                    extraction_type = "range"
                    field_conf = 0.75  # Medium confidence for ranges
                elif field_meta and field_meta.is_approximate:
                    extraction_type = "approximate"
                    field_conf = 0.70  # Lower for approximates
                else:
                    extraction_type = "exact"
                    field_conf = 0.95  # High for exact values

                # Adjust based on quality
                if extraction_quality == "low":
                    field_conf -= 0.10
                elif extraction_quality == "medium":
                    field_conf -= 0.05

                # Extra reduction for critical fields that are approximate
                if field in self.CRITICAL_FIELDS and extraction_type != "exact":
                    field_conf -= 0.10

                field_conf = max(0.0, min(1.0, field_conf))

                field_confidences[field] = {
                    "confidence": round(field_conf, 4),
                    "extraction_type": extraction_type,
                    "level": "high" if field_conf >= 0.8 else ("medium" if field_conf >= 0.5 else "low"),
                }

        return {
            "overall_confidence": round(overall_confidence, 4),
            "confidence_level": confidence_level,
            "field_confidences": field_confidences,
            "quality_artifacts": {
                "approx_fields": approx_fields,
                "range_fields": range_fields,
                "missing_fields": missing_fields,
                "critical_approx": critical_approx,
                "critical_range": critical_range,
                "high_approximate_ratio": high_approx_ratio,
                "extraction_quality": extraction_quality,
                "base_score": round(base_score, 4),
                "penalties_applied": round(penalties, 4),
            },
        }


class AIBrainPipeline:
    def __init__(self, storage_dir: Path | None = None) -> None:
        self.ingestion = IngestionService(storage_dir=storage_dir or DEFAULT_STORAGE_DIR)
        self.extraction = ExtractionService()
        self.vectorless_rag = VectorlessRAGExtractor()
        self.rule_engine = RuleEngine()
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.insight_generator = FinancialInsightGenerator()

    def run(self, company: str, file_paths: List[str | Path], include_metadata: bool = False) -> Dict[str, Any]:
        """
        Execute the AIBrain pipeline once with structured output.

        Returns structured result with financials, insights, warnings, and confidence metrics.
        """
        try:
            ingested_files = self.ingestion.ingest_files(file_paths)
            raw_text = self.extraction.extract_many(ingested_files)
        except Exception:
            return {
                "status": "failed",
                "reason": "No usable financial data",
                "warnings": [],
            }

        if not raw_text.strip():
            return {
                "status": "failed",
                "reason": "No usable financial data",
                "warnings": [],
            }

        # Extract financials and metadata
        financials, metadata = self.vectorless_rag.extract_financials(raw_text)
        extracted_fields = sum(1 for key in FINANCIAL_FIELDS if financials.get(key) is not None)

        if extracted_fields == 0:
            return {
                "status": "failed",
                "reason": "No usable financial data",
                "warnings": [],
            }

        # Check if we have minimum fields
        null_count = len(FINANCIAL_FIELDS) - extracted_fields
        if null_count > len(FINANCIAL_FIELDS) / 2:
            return {
                "status": "insufficient_data",
                "confidence": 0.0,
                "confidence_level": "low",
                "warnings": ["More than 50% of financial fields could not be extracted"],
            }

        # Evaluate rules and extract quality
        rule_result = self.rule_engine.evaluate(financials)
        extraction_quality = self._detect_extraction_quality(raw_text)
        resolved_company = self._resolve_company_name(company, raw_text)

        # Calculate confidence with detailed metrics
        confidence_metrics = self.confidence_analyzer.calculate(
            financials=financials,
            metadata=metadata,
            flags=rule_result.flags,
            extraction_quality=extraction_quality
        )

        # Generate insights and warnings
        insights = self.insight_generator.generate(
            financials=financials,
            diagnostics=rule_result.diagnostics,
            extraction_metadata=confidence_metrics.get("quality_artifacts")
        )

        # Build decision-ready deterministic result
        ratios = self._build_ratios(financials)
        structured_risk_flags = self._build_structured_risk_flags(ratios)
        field_confidence = self._build_field_confidence(financials, metadata, extraction_quality)
        data_quality_score = self._build_data_quality_score(field_confidence)
        calibrated_confidence = data_quality_score
        confidence_level = self._confidence_level_from_score(calibrated_confidence)

        qualitative_flags = self._build_qualitative_risk_flags(raw_text, structured_risk_flags)
        all_risk_flags = self._merge_risk_flags(structured_risk_flags, qualitative_flags)
        risk_contributions = self._build_risk_contributions(all_risk_flags)
        key_risk_drivers = self._build_key_risk_drivers(all_risk_flags)
        insights = self._ensure_insight_coverage(all_risk_flags, insights, raw_text)
        base_score = 100
        final_score = self._build_final_score(base_score, risk_contributions)

        result = {
            "company": resolved_company,
            "financials": financials,
            "ratios": ratios,
            "risk_flags": all_risk_flags,
            "risk_contributions": risk_contributions,
            "key_risk_drivers": key_risk_drivers,
            "insights": insights,
            "field_confidence": field_confidence,
            "data_quality_score": data_quality_score,
            "confidence": calibrated_confidence,
            "confidence_level": confidence_level,
            "base_score": base_score,
            "final_score": final_score,
            "warnings": [self._consolidate_warning(financials, field_confidence)],
        }

        # STEP FINAL: strict correction/validation pass to guarantee internal consistency.
        result = self._validate_and_correct_result(result, raw_text)

        # Optionally include detailed metadata
        if include_metadata:
            result["_metadata"] = {
                "field_metrics": confidence_metrics["field_confidences"],
                "quality_artifacts": confidence_metrics["quality_artifacts"],
                "field_details": {
                    field: {
                        "confidence_bin": meta.confidence if meta else None,
                        "is_approximate": meta.is_approximate if meta else None,
                        "has_range": meta.has_range if meta else None,
                        "min": meta.min_value if meta else None,
                        "max": meta.max_value if meta else None,
                    }
                    for field, meta in metadata.items()
                    if meta is not None
                },
                "source_evidence": self._build_source_evidence(
                    ingested_files,
                    self.vectorless_rag.last_evidence,
                ),
            }

        return result

    def _build_source_evidence(
        self,
        file_paths: List[str | Path],
        evidence_by_field: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        """Map supporting text to a unique source segment without guessing."""
        result: Dict[str, Dict[str, Any]] = {}

        try:
            documents = self.extraction.extract_documents(
                [Path(file_path) for file_path in file_paths]
            )
        except Exception:
            documents = []

        normalized_documents = []
        for document in documents:
            text = normalize_whitespace(
                str(document.get("text") or "")
            ).casefold()
            normalized_documents.append((document, text))

        for field in FINANCIAL_FIELDS:
            evidence = evidence_by_field.get(field, "")
            entry: Dict[str, Any] = {
                "evidence_text": evidence or None,
                "source": None,
                "page": None,
                "status": "no_evidence",
            }

            if not evidence:
                result[field] = entry
                continue

            normalized_evidence = normalize_whitespace(evidence).casefold()

            # Numeric-only or very short snippets are not reliable provenance.
            if len(normalized_evidence) < 12 or not re.search(
                r"[a-z]", normalized_evidence
            ):
                entry["status"] = "insufficient_context"
                result[field] = entry
                continue

            matches = [
                document
                for document, document_text in normalized_documents
                if normalized_evidence in document_text
            ]

            if len(matches) == 1:
                entry["source"] = matches[0].get("source")
                entry["page"] = matches[0].get("page")
                entry["status"] = (
                    "matched"
                    if entry["source"] and entry["page"] is not None
                    else "matched_page_unknown"
                )
            elif len(matches) > 1:
                entry["status"] = "ambiguous"
            else:
                entry["status"] = "not_found"

            result[field] = entry

        return result

    def _resolve_company_name(self, provided_company: str, raw_text: str) -> str:
        extracted = self._extract_company_name(raw_text)
        return extracted if extracted else provided_company

    def _extract_company_name(self, raw_text: str) -> str | None:
        if not raw_text:
            return None

        company_label_pattern = re.compile(
            r"(?i)(?:company(?:_name|\s+name)?|entity)\s*[:=]\s*([A-Za-z0-9&.,()'\- ]{3,80})"
        )
        for line in raw_text.splitlines():
            line_match = company_label_pattern.search(line)
            if not line_match:
                continue
            candidate = self._clean_company_candidate(line_match.group(1))
            if candidate and re.search(r"[A-Za-z]", candidate):
                return candidate

        suffix_pattern = re.compile(
            r"\b(?:pvt\.?|private|ltd\.?|limited|llp|inc\.?|corp\.?|corporation|company|co\.?)\b",
            re.IGNORECASE,
        )
        heading_tokens = {
            "unaudited financial snapshot",
            "balance sheet",
            "auditor remarks",
            "management notes",
            "additional notes",
            "financial snapshot",
            "cash flow",
            "revenue",
            "gst",
            "receivables",
            "borrowings",
            "inventory",
        }

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for line in lines[:15]:
            candidate = self._clean_company_candidate(line)
            lower_candidate = candidate.lower()

            if len(candidate) < 3:
                continue
            if candidate.startswith("("):
                continue
            if any(token in lower_candidate for token in heading_tokens):
                continue
            if suffix_pattern.search(candidate) and re.search(r"[A-Za-z]", candidate):
                return candidate

        return None

    def _clean_company_candidate(self, candidate: str) -> str:
        cleaned = re.sub(r"\s+", " ", candidate).strip("-|:\t \n\r")
        # Guard against flattened key/value tails like "... PVT LTD revenue: 100".
        cleaned = re.sub(r"\s+[a-z_][a-z0-9_ ]{0,30}\s*:\s*.*$", "", cleaned, flags=re.IGNORECASE)
        suffix_pattern = re.compile(
            r"\b(?:pvt\.?|private|ltd\.?|limited|llp|inc\.?|corp\.?|corporation|company|co\.?)\b",
            re.IGNORECASE,
        )
        suffix_matches = list(suffix_pattern.finditer(cleaned))
        if suffix_matches:
            cleaned = cleaned[:suffix_matches[-1].end()]
        return cleaned.strip("-|:\t \n\r")

    def _safe_ratio(self, numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or denominator == 0:
            return None
        return round(numerator / denominator, 2)

    def _build_ratios(self, financials: Dict[str, float | None]) -> Dict[str, float | None]:
        return {
            "cash_flow_to_profit": self._safe_ratio(
                financials.get("operating_cash_flow"), financials.get("net_profit")
            ),
            "debt_to_revenue": self._safe_ratio(
                financials.get("total_debt"), financials.get("revenue")
            ),
            "receivables_to_revenue": self._safe_ratio(
                financials.get("receivables"), financials.get("revenue")
            ),
            "bank_to_gst_ratio": self._safe_ratio(
                financials.get("bank_deposits"), financials.get("gst_revenue")
            ),
        }

    def _build_structured_risk_flags(self, ratios: Dict[str, float | None]) -> List[Dict[str, Any]]:
        risk_flags: List[Dict[str, Any]] = []

        bank_to_gst = ratios.get("bank_to_gst_ratio")
        if bank_to_gst is not None and bank_to_gst < 0.75:
            risk_flags.append(
                {
                    "type": "revenue_mismatch",
                    "severity": "high",
                    "score_impact": -25,
                    "metric": "bank_to_gst_ratio",
                    "value": bank_to_gst,
                    "threshold": "< 0.75",
                }
            )

        cash_flow_to_profit = ratios.get("cash_flow_to_profit")
        if cash_flow_to_profit is not None and cash_flow_to_profit < 0.7:
            risk_flags.append(
                {
                    "type": "low_cash_flow_vs_profit",
                    "severity": "high",
                    "score_impact": -20,
                    "metric": "cash_flow_to_profit",
                    "value": cash_flow_to_profit,
                    "threshold": "< 0.7",
                }
            )

        receivables_to_revenue = ratios.get("receivables_to_revenue")
        if receivables_to_revenue is not None and receivables_to_revenue > 0.5:
            risk_flags.append(
                {
                    "type": "high_receivables",
                    "severity": "medium",
                    "score_impact": -15,
                    "metric": "receivables_to_revenue",
                    "value": receivables_to_revenue,
                    "threshold": "> 0.5",
                }
            )

        debt_to_revenue = ratios.get("debt_to_revenue")
        if debt_to_revenue is not None and debt_to_revenue > 0.7:
            risk_flags.append(
                {
                    "type": "high_leverage",
                    "severity": "high",
                    "score_impact": -20,
                    "metric": "debt_to_revenue",
                    "value": debt_to_revenue,
                    "threshold": "> 0.7",
                }
            )

        return risk_flags

    def _build_key_risk_drivers(self, risk_flags: List[Dict[str, Any]]) -> List[str]:
        risk_driver_map = {
            "revenue_mismatch": "Revenue realization gap",
            "low_cash_flow_vs_profit": "Weak operating cash conversion",
            "high_receivables": "Elevated receivables pressure",
            "high_leverage": "Leverage burden",
            "governance_risk": "Related-party governance concern",
        }

        drivers: List[str] = []
        for flag in risk_flags:
            risk_type = flag.get("type")
            if not isinstance(risk_type, str):
                continue
            driver = risk_driver_map.get(risk_type, f"{risk_type.replace('_', ' ').title()} risk")
            if driver not in drivers:
                drivers.append(driver)

        return drivers[:5]

    def _build_risk_contributions(self, risk_flags: List[Dict[str, Any]]) -> Dict[str, int]:
        return {
            flag["type"]: flag["score_impact"]
            for flag in risk_flags
            if "type" in flag and "score_impact" in flag
        }

    def _build_field_confidence(
        self,
        financials: Dict[str, float | None],
        metadata: Dict[str, Any],
        extraction_quality: Literal["high", "medium", "low"],
    ) -> Dict[str, str]:
        def downgrade(level: str, steps: int) -> str:
            levels = ["high", "medium", "low"]
            index = levels.index(level)
            return levels[min(len(levels) - 1, index + steps)]

        confidence_by_field: Dict[str, str] = {}
        for field in FINANCIAL_FIELDS:
            value = financials.get(field)
            if value is None:
                confidence_by_field[field] = "low"
                continue

            field_meta = metadata.get(field)
            if field_meta and field_meta.has_range and field_meta.is_approximate:
                base_confidence = "low"
            elif field_meta and field_meta.has_range:
                base_confidence = "medium"
            elif field_meta and field_meta.is_approximate:
                base_confidence = "low"
            else:
                base_confidence = "high"

            if extraction_quality == "medium" and base_confidence != "high":
                base_confidence = downgrade(base_confidence, 1)
            elif extraction_quality == "low":
                base_confidence = downgrade(base_confidence, 1)

            confidence_by_field[field] = base_confidence

        return confidence_by_field

    def _build_data_quality_score(self, field_confidence: Dict[str, str]) -> float:
        mapping = {"high": 1.0, "medium": 0.75, "low": 0.5}
        score_components = [mapping.get(field_confidence.get(field, "low"), 0.5) for field in FINANCIAL_FIELDS]
        if not score_components:
            return 0.0
        return round(sum(score_components) / len(score_components), 2)


    def _consolidate_warning(
        self,
        financials: Dict[str, float | None],
        field_confidence: Dict[str, str],
    ) -> str:
        missing_count = sum(
            1
            for field in FINANCIAL_FIELDS
            if financials.get(field) is None
        )

        low_count = sum(
            1
            for field in FINANCIAL_FIELDS
            if field_confidence.get(field) == "low"
            and financials.get(field) is not None
        )

        medium_count = sum(
            1
            for field in FINANCIAL_FIELDS
            if field_confidence.get(field) == "medium"
            and financials.get(field) is not None
        )

        quality_score = self._build_data_quality_score(field_confidence)

        if quality_score > 0.75:
            reliability = "high"
        elif quality_score >= 0.60:
            reliability = "moderate"
        else:
            reliability = "low"

        return (
            f"Extraction-based data reliability is {reliability} "
            f"(heuristic score: {quality_score:.2f}). "
            f"{low_count} extracted field(s) have low confidence, "
            f"{medium_count} extracted field(s) have medium confidence, "
            f"and {missing_count} field(s) are missing. "
            "Validate extracted figures against finalized financial "
            "statements before making credit decisions."
        )


    def _build_qualitative_risk_flags(self, raw_text: str, existing_flags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text = (raw_text or "").lower()
        existing_types = {flag.get("type") for flag in existing_flags if flag.get("type")}
        flags: List[Dict[str, Any]] = []

        def add_if_missing(flag_type: str, severity: str, score_impact: int, threshold: str) -> None:
            if flag_type in existing_types:
                return
            flags.append(
                {
                    "type": flag_type,
                    "severity": severity,
                    "score_impact": score_impact,
                    "metric": "qualitative_signal",
                    "value": 1.0,
                    "threshold": threshold,
                }
            )

        if re.search(r"related\s+part(?:y|ies)", text):
            add_if_missing("governance_risk", "high", -20, "keyword: related party")

        if re.search(r"cash\s*flow[^\n]{0,40}stress|cash\s*flow\s*position\s*appears\s*stressed", text):
            add_if_missing("low_cash_flow_vs_profit", "high", -20, "keyword: cash flow stressed")

        if re.search(r"receivables?[^\n]{0,40}delay|delays\s+in\s+receivables", text):
            add_if_missing("high_receivables", "medium", -15, "keyword: receivables delay")

        if re.search(r"bank[^\n]{0,40}mismatch|bank\s+inflows\s+do\s+not\s+fully\s+align", text):
            add_if_missing("revenue_mismatch", "high", -25, "keyword: bank mismatch")

        return flags

    def _merge_risk_flags(self, quantitative: List[Dict[str, Any]], qualitative: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged = list(quantitative)
        existing = {flag.get("type") for flag in quantitative if flag.get("type")}
        for flag in qualitative:
            if flag.get("type") not in existing:
                merged.append(flag)
        return merged

    def _ensure_insight_coverage(
        self, risk_flags: List[Dict[str, Any]], base_insights: List[str], raw_text: str
    ) -> List[str]:
        insights = list(base_insights)
        risk_to_insight = {
            "revenue_mismatch": "The gap between GST turnover and bank inflows may indicate slower realization cycles and could lead to working capital pressure.",
            "low_cash_flow_vs_profit": "Operating cash conversion is weaker than reported profitability, which may indicate accrual-heavy earnings and liquidity strain.",
            "high_receivables": "High receivables intensity suggests potential collection delays and may indicate elevated counterparty payment risk.",
            "high_leverage": "Debt relative to revenue remains elevated, which may constrain financing flexibility and increase refinancing vulnerability.",
            "governance_risk": "Related-party transaction remarks may indicate governance concerns and could lead to heightened monitoring requirements.",
        }

        present = set()
        for insight in insights:
            lower = insight.lower()
            if "gst" in lower or "bank" in lower:
                present.add("revenue_mismatch")
            if "cash flow" in lower and ("profit" in lower or "earnings" in lower):
                present.add("low_cash_flow_vs_profit")
            if "receivable" in lower:
                present.add("high_receivables")
            if "debt" in lower or "leverage" in lower:
                present.add("high_leverage")
            if "related-party" in lower or "governance" in lower:
                present.add("governance_risk")

        for flag in risk_flags:
            risk_type = flag.get("type")
            if risk_type not in present and risk_type in risk_to_insight:
                insights.append(risk_to_insight[risk_type])
                present.add(str(risk_type))

        return insights

    def _build_final_score(self, base_score: int, risk_contributions: Dict[str, int]) -> int:
        final_score = base_score + sum(risk_contributions.values())
        return max(0, min(100, final_score))

    def _confidence_level_from_score(self, score: float) -> str:
        if score > 0.75:
            return "high"
        if score >= 0.6:
            return "medium"
        return "low"

    def _validate_and_correct_result(self, result: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        financials: Dict[str, float | None] = result.get("financials", {})

        # Recompute ratios from final financial values.
        ratios = self._build_ratios(financials)

        # Recompute risk flags and contributions directly from ratios.
        risk_flags = self._build_structured_risk_flags(ratios)
        qualitative_flags = self._build_qualitative_risk_flags(raw_text, risk_flags)
        risk_flags = self._merge_risk_flags(risk_flags, qualitative_flags)
        risk_contributions = self._build_risk_contributions(risk_flags)
        key_risk_drivers = self._build_key_risk_drivers(risk_flags)

        # Ensure risk flag metric values exactly match ratio values.
        metric_to_ratio = {
            "cash_flow_to_profit": ratios.get("cash_flow_to_profit"),
            "debt_to_revenue": ratios.get("debt_to_revenue"),
            "receivables_to_revenue": ratios.get("receivables_to_revenue"),
            "bank_to_gst_ratio": ratios.get("bank_to_gst_ratio"),
        }
        for flag in risk_flags:
            metric = flag.get("metric")
            if metric in metric_to_ratio:
                flag["value"] = metric_to_ratio[metric]

        # Recompute warning from current confidence and missing counts.
        field_confidence: Dict[str, str] = result.get("field_confidence", {})
        warnings = [self._consolidate_warning(financials, field_confidence)]

        # Recompute data quality score from scratch.
        data_quality_score = self._build_data_quality_score(field_confidence)
        confidence = data_quality_score
        confidence_level = self._confidence_level_from_score(confidence)
        base_score = 100
        final_score = self._build_final_score(base_score, risk_contributions)
        insights = self._ensure_insight_coverage(risk_flags, result.get("insights", []), raw_text)

        result["ratios"] = ratios
        result["risk_flags"] = risk_flags
        result["risk_contributions"] = risk_contributions
        result["key_risk_drivers"] = key_risk_drivers
        result["insights"] = insights
        result["data_quality_score"] = data_quality_score
        result["confidence"] = confidence
        result["confidence_level"] = confidence_level
        result["base_score"] = base_score
        result["final_score"] = final_score
        result["warnings"] = warnings

        return result

    def _detect_extraction_quality(self, text: str) -> Literal["high", "medium", "low"]:
        """
        Infer extraction quality from text characteristics.
        Returns: high, medium, or low
        """
        if not text:
            return "low"

        # Count various noise indicators.
        lines = text.split('\n')
        total_lines = max(1, len(lines))

        short_lines = sum(1 for line in lines if len(line.strip().split()) <= 2)
        short_line_ratio = short_lines / total_lines

        symbol_only_lines = sum(1 for line in lines if line.strip() and not re.search(r"[A-Za-z0-9]", line))
        symbol_line_ratio = symbol_only_lines / total_lines

        ocr_artifacts = len(re.findall(r"[|!{}\[\]`^¯ˆ~¨¡§¶†]", text))
        char_count = max(1, len(text))
        artifact_ratio = ocr_artifacts / char_count

        confusable_pairs = len(re.findall(r"\d[OoIlSsbB]|[OoIlSsbB]\d", text))
        confusable_ratio = confusable_pairs / max(1, total_lines)

        broken_words = len(re.findall(r"(?:^|\s)[a-z]+-[a-z]+-", text, re.MULTILINE))
        broken_word_ratio = broken_words / total_lines

        noise_ratio = (
            0.35 * short_line_ratio
            + 0.20 * symbol_line_ratio
            + 2.5 * artifact_ratio
            + 0.20 * confusable_ratio
            + 0.25 * broken_word_ratio
        )

        # Classify quality
        if noise_ratio < 0.12:
            return "high"
        elif noise_ratio < 0.28:
            return "medium"
        else:
            return "low"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AI Brain deterministic credit decisioning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--company", required=True, help="Company name")
    parser.add_argument("--files", nargs="+", required=True, help="Input files (PDF/CSV/JSON)")
    parser.add_argument(
        "--storage-dir",
        default=str(DEFAULT_STORAGE_DIR),
        help="Directory to store ingested files"
    )
    return parser


def main() -> None:
    """Execute the AIBrain pipeline with clean, structured output."""
    parser = build_parser()
    args = parser.parse_args()

    pipeline = AIBrainPipeline(storage_dir=Path(args.storage_dir))
    result = pipeline.run(company=args.company, file_paths=args.files)

    # Output structured JSON once without debug messages
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
