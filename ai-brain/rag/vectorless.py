"""Strict vectorless extractor for explicit financial fields only."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Dict, Literal, Optional

from config.settings import FINANCIAL_FIELDS
from models.schemas import Financials, FinancialValue
from utils.text_utils import (
    UNIT_TOKEN_PATTERN,
    normalize_whitespace,
    parse_approximate_value,
)

FIELD_PATTERNS: Dict[str, list[str]] = {
    "revenue": ["total revenue", "revenue", "sales", "turnover"],
    "net_profit": ["net profit", "net profit after tax", "profit after tax", "pat"],
    "total_debt": ["total debt", "debt", "borrowings", "total liabilities"],
    "bank_deposits": ["bank deposits", "bank inflows", "cash inflow", "bank credits", "deposits in bank", "bank balance", "bank collections"],
    "gst_revenue": ["gst declared revenue", "gst revenue", "gst turnover", "gst filings", "gstr revenue", "gst sales"],
    "receivables": ["trade receivables", "accounts receivable", "receivables", "debtors"],
    "operating_cash_flow": ["operating cash flow", "cash flow from operations", "cash flow from ops", "cfo", "net cash from operations"],
}

FINANCIAL_SECTION_HINTS = {
    "financial",
    "statement",
    "balance sheet",
    "p&l",
    "profit",
    "loss",
    "cash flow",
    "gstr",
    "gst",
    "turnover",
}

# Approximate indicators for confidence reduction
APPROXIMATE_INDICATORS = r"~|approx(?:imately)?|around|circa|about|roughly|~?(?:approx|est\.?)"
APPROXIMATE_LINE_PATTERN = re.compile(
    r"(?:~|\bapprox(?:\.|imately)?\b|\baround\b|\babout\b|\best(?:\.|imated)?\b|\broughly\b|\bcirca\b)",
    re.IGNORECASE,
)

NEGATION_PATTERN = re.compile(r"\b(?:no|nil|none|without|zero|not applicable|n/?a)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    value: float
    score: int
    confidence: Literal["high", "medium", "low"] = "high"
    is_approximate: bool = False
    has_range: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    value_type: str = "exact"  # exact | approximate | range


def _normalize_alias(alias: str) -> str:
    return normalize_whitespace(alias.lower())


def _searchable_text(text: str) -> str:
    # Keep length stable so match positions remain usable for proximity scoring.
    normalized = text.lower()
    # Common PDF ligatures that break keyword matching (e.g., proﬁt, ﬁlings).
    normalized = (
        normalized.replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
        .replace("ﬀ", "ff")
        .replace("ﬃ", "ffi")
        .replace("ﬄ", "ffl")
    )
    return normalized.replace("_", " ")


class VectorlessRAGExtractor:
    def extract_financials(self, raw_text: str) -> tuple[Financials, Dict]:
        """
        Extract financial fields and return both the simple dict and metadata.
        Returns: (financials_dict, metadata_dict)
        """
        output: Financials = {field: None for field in FINANCIAL_FIELDS}
        metadata: Dict = {field: None for field in FINANCIAL_FIELDS}

        for field, aliases in FIELD_PATTERNS.items():
            candidate = self._extract_field_value(raw_text, field, aliases)
            if candidate:
                output[field] = candidate.value
                metadata[field] = FinancialValue(
                    value=candidate.value,
                    confidence=candidate.confidence,
                    is_approximate=candidate.is_approximate,
                    has_range=candidate.has_range,
                    min_value=candidate.min_value,
                    max_value=candidate.max_value,
                )

        return output, metadata

    def _extract_field_value(self, text: str, field: str, aliases: list[str]) -> Optional[Candidate]:
        normalized_text = normalize_whitespace(_searchable_text(text))
        normalized_lines = []
        for raw_line in text.splitlines():
            line = normalize_whitespace(raw_line)
            if not line:
                continue

            # Split sentences merged together by PDF extraction.
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
            normalized_lines.extend(
                sentence.strip()
                for sentence in sentences
                if sentence.strip()
            )
        candidates: list[Candidate] = []

        normalized_aliases = [_normalize_alias(alias) for alias in aliases]

        for idx, line in enumerate(normalized_lines):
            line_candidates = self._extract_from_line(line, field, aliases)
            candidates.extend(line_candidates)

            # Support formats where label is on one line and value is on the next line.
            line_has_alias = any(alias in _searchable_text(line) for alias in normalized_aliases)
            line_has_number = bool(re.search(r"\d", line))
            if idx + 1 < len(normalized_lines) and line_has_alias and not line_has_number:
                window_line = normalize_whitespace(f"{line} {normalized_lines[idx + 1]}")
                window_candidates = self._extract_from_line(window_line, field, aliases)
                candidates.extend(window_candidates)

        if candidates:
            def has_material_conflict(selected: Candidate, all_candidates: list[Candidate]) -> bool:
                baseline = abs(selected.value) if selected.value != 0 else 1.0
                for candidate in all_candidates:
                    if candidate is selected:
                        continue
                    if abs(candidate.value - selected.value) / baseline > 0.2:
                        return True
                return False

            range_candidates = [c for c in candidates if c.has_range]

            if range_candidates:
                best_range = max(range_candidates, key=lambda c: c.score)

                # Preserve range priority, but flag conflicts with other extracted values.
                approx_candidates = [
                    c for c in candidates
                    if c.is_approximate and not c.has_range
                ]

                if approx_candidates and not best_range.is_approximate:
                    best_range = replace(
                        best_range,
                        is_approximate=True,
                        confidence="low",
                    )

                if has_material_conflict(best_range, candidates):
                    best_range = replace(
                        best_range,
                        is_approximate=True,
                        confidence="low",
                    )

                return best_range

            best_candidate = max(candidates, key=lambda c: c.score)
            if has_material_conflict(best_candidate, candidates):
                best_candidate = replace(best_candidate, is_approximate=True, confidence="low")
            return best_candidate

        # Fallback: scan entire text with assignment-bound pattern only.
        for alias in aliases:
            alias_pattern = re.escape(_normalize_alias(alias)).replace(r"\ ", r"\s+")

            fallback_patterns = [
                rf"(?:{alias_pattern})\s*(?:[:=\-]|is)\s*(?:~|\b(?:approx|around|about|estimated|est\.?))?\s*(?:rs\.?|inr)?\s*(\(?[-+]?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:{UNIT_TOKEN_PATTERN}))?(?:\s*(?:[-–—]|to)\s*\(?[-+]?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:{UNIT_TOKEN_PATTERN}))?)?)",
            ]

            for pattern in fallback_patterns:
                best_match = None
                best_score = -1

                for match in re.finditer(pattern, normalized_text, flags=re.IGNORECASE):
                    matched_text = match.group(1)
                    approx_info = parse_approximate_value(matched_text)

                    if not approx_info:
                        continue

                    if self._is_year_like_value(approx_info['value']):
                        continue

                    # Prefer explicit range and tight assignment context.
                    match_score = 10
                    if approx_info['has_range']:
                        match_score += 20

                    if match_score > best_score:
                        best_score = match_score
                        best_match = approx_info

                if best_match:
                    value_type = "range" if best_match["has_range"] else ("approximate" if best_match["is_approximate"] else "exact")
                    fallback_conf: Literal["high", "medium", "low"] = (
                        best_match["confidence"]
                        if best_match.get("confidence") in ("high", "medium", "low")
                        else "high"
                    )
                    return Candidate(
                        value=best_match['value'],
                        score=50,  # Lower score for fallback
                        confidence=fallback_conf,
                        is_approximate=best_match['is_approximate'],
                        has_range=best_match['has_range'],
                        min_value=best_match.get('min'),
                        max_value=best_match.get('max'),
                        value_type=value_type,
                    )

        return None

    def _extract_from_line(self, line: str, field: str, aliases: list[str]) -> list[Candidate]:
        line_searchable = _searchable_text(line)
        line_lower = line_searchable
        results: list[Candidate] = []

        # Normalize the line for consistent pattern matching
        normalized_line = normalize_whitespace(line_lower)

        # Patterns to find numbers:
        # 1. Ranges with explicit connector: "60 to 65" / "2 Cr to 3 Cr"
        # 2. Ranges with explicit dash connectors: "60-65"
        # 3. Approximate: "around 100" or "~100"
        number_token = r"(?:\(?[-+]?\d[\d,]*(?:\.\d+)?\)?)"
        range_with_connector = (
            rf"({number_token})"
            rf"(?:\s*(?:{UNIT_TOKEN_PATTERN}))?"
            r"\s+to\s+"
            rf"({number_token})"
            rf"(?:\s*(?:{UNIT_TOKEN_PATTERN}))?"
        )
        range_with_dash = (
            rf"({number_token})"
            rf"(?:\s*(?:{UNIT_TOKEN_PATTERN}))?"
            r"\s*(?:[-–—])\s*"
            rf"({number_token})"
            rf"(?:\s*(?:{UNIT_TOKEN_PATTERN}))?"
        )
        approx_with_number = rf"(?:~|approx(?:\.|imately)?|around|ca(?:\.|$)|est(?:\.|imated)?|about|roughly|estimated)\s*(?:rs\.?|inr)?\s*({number_token}(?:\s*(?:{UNIT_TOKEN_PATTERN}))?)"
        exact_number_pattern = rf"(?<![a-z0-9])({number_token}(?:\s*(?:{UNIT_TOKEN_PATTERN}))?)(?![a-z0-9])"

        number_matches = []

        # Extract ranges with explicit connector first (e.g., 60 to 65)
        for match in re.finditer(range_with_connector, normalized_line, re.IGNORECASE):
            val1_str = match.group(1)
            val2_str = match.group(2)
            try:
                val1_info = parse_approximate_value(val1_str)
                val2_info = parse_approximate_value(val2_str)
                val1 = val1_info.get("value") if val1_info else None
                val2 = val2_info.get("value") if val2_info else None
                if val1 is None or val2 is None:
                    continue
                ratio = abs(val2 / val1) if val1 != 0 else 0
                if 0.5 < ratio < 2:
                    match_text = f"{val1_str} to {val2_str}"
                    number_matches.append((match, match_text, 'range'))
            except ValueError:
                pass

        # Extract ranges with explicit dash connector.
        for match in re.finditer(range_with_dash, normalized_line):
            val1_str = match.group(1)
            val2_str = match.group(2)
            try:
                val1_info = parse_approximate_value(val1_str)
                val2_info = parse_approximate_value(val2_str)
                val1 = val1_info.get("value") if val1_info else None
                val2 = val2_info.get("value") if val2_info else None
                if val1 is None or val2 is None:
                    continue
                ratio = abs(val2 / val1) if val1 != 0 else 0
                if 0.5 < ratio < 2:
                    match_text = f"{val1_str}-{val2_str}"
                    number_matches.append((match, match_text, 'range'))
            except ValueError:
                pass

        # Extract approximate indicators with numbers
        for match in re.finditer(approx_with_number, normalized_line, re.IGNORECASE):
            number_matches.append((match, match.group(1), 'approx'))

        # Extract exact numbers only when they do not overlap with approx/range matches.
        protected_spans = [(m.start(), m.end()) for m, _, t in number_matches if t in {"range", "approx"}]
        for match in re.finditer(exact_number_pattern, normalized_line):
            m_start, m_end = match.start(), match.end()
            overlaps = any(not (m_end <= s or m_start >= e) for s, e in protected_spans)
            if overlaps:
                continue
            exact_token = match.group(1)
            exact_info = parse_approximate_value(exact_token)
            if not exact_info:
                continue
            if self._is_year_like_value(exact_info["value"]):
                continue
            number_matches.append((match, exact_token, 'exact'))

        if not number_matches:
            return results

        is_table_like = any(token in line for token in ("|", ";", "\t"))
        has_financial_context = any(hint in line_lower for hint in FINANCIAL_SECTION_HINTS)

        for alias in aliases:
            alias_normalized = _normalize_alias(alias)
            alias_match = re.search(re.escape(alias_normalized), normalized_line)
            if not alias_match:
                continue

            if self._is_negated_alias(normalized_line, alias_normalized):
                results.append(
                    Candidate(
                        value=0.0,
                        score=150 + len(alias_normalized),
                        confidence="high",
                        is_approximate=False,
                        has_range=False,
                        value_type="exact",
                    )
                )
                continue

            alias_index = alias_match.start()
            line_has_approx_indicator = bool(APPROXIMATE_LINE_PATTERN.search(normalized_line))

            for match, match_text, match_type in number_matches:
                # Try to parse as approximate value (handles ranges, ~, etc)
                approx_info = parse_approximate_value(match_text)
                if not approx_info:
                    continue

                # If line wording signals approximation but value token looks exact,
                # enforce low confidence per source wording rules.
                if line_has_approx_indicator and not approx_info["has_range"]:
                    approx_info["is_approximate"] = True
                    approx_info["confidence"] = "low"

                parsed = approx_info['value']
                if self._is_year_like_value(parsed):
                    continue

                # Adjust score based on confidence
                confidence_penalty = 0 if approx_info['confidence'] == 'high' else 20

                # Bonus for range matches
                range_bonus = 30 if approx_info['has_range'] else 0

                distance = abs(match.start() - alias_index)
                score = 100 - min(distance, 90) - confidence_penalty + range_bonus

                if match.start() >= alias_index:
                    score += 20
                else:
                    score -= 15

                # Check for direct assignment pattern
                if match.start() >= alias_index and distance <= 25 and re.search(
                    rf"{re.escape(alias_normalized)}\s*(?:[:=\-]|is)\s*(?:rs\.?|inr)?\s*(?:~)?",
                    normalized_line,
                    flags=re.IGNORECASE,
                ):
                    score += 45

                # Disambiguation: avoid treating GST turnover lines as plain revenue.
                if field == "revenue" and re.search(r"\bgst|gstr\b", line_lower, flags=re.IGNORECASE):
                    score -= 40
                if field == "gst_revenue" and re.search(r"\bgst|gstr\b", line_lower, flags=re.IGNORECASE):
                    score += 40

                if is_table_like:
                    score += 15
                if has_financial_context:
                    score += 10

                # Prefer more specific aliases
                score += len(alias_normalized)

                value_type = "range" if approx_info['has_range'] else ("approximate" if approx_info['is_approximate'] else "exact")
                match_conf: Literal["high", "medium", "low"] = (
                    approx_info["confidence"]
                    if approx_info.get("confidence") in ("high", "medium", "low")
                    else "high"
                )
                results.append(
                    Candidate(
                        value=parsed,
                        score=score,
                        confidence=match_conf,
                        is_approximate=approx_info['is_approximate'],
                        has_range=approx_info['has_range'],
                        min_value=approx_info.get('min'),
                        max_value=approx_info.get('max'),
                        value_type=value_type,
                    )
                )

        return results

    def _is_negated_alias(self, normalized_line: str, alias: str) -> bool:
        if not NEGATION_PATTERN.search(normalized_line):
            return False
        if re.search(rf"\b(?:no|nil|none|without|zero)\b\s+(?:\w+\s+){{0,2}}{re.escape(alias)}\b", normalized_line):
            return True
        if re.search(rf"\b{re.escape(alias)}\b\s*(?:[:=]|is)?\s*(?:nil|none|no|zero|n/?a)\b", normalized_line):
            return True
        return False

    def _is_year_like_value(self, value: float) -> bool:
        integer_value = round(value)
        return abs(value - integer_value) < 1e-9 and 1900 <= integer_value <= 2100
