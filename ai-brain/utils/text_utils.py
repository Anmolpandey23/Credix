"""Text parsing utilities with strict non-hallucinating behavior."""

import re
from typing import Optional, Dict


UNIT_MULTIPLIERS_TO_CRORE = {
    "cr": 1.0,
    "crore": 1.0,
    "crores": 1.0,
    "mn": 0.1,
    "million": 0.1,
    "m": 0.1,
    "bn": 100.0,
    "billion": 100.0,
    "lakh": 0.01,
    "lakhs": 0.01,
    "lac": 0.01,
    "lacs": 0.01,
}

UNIT_TOKEN_PATTERN = r"(?:cr|crores?|mn|million|bn|billion|lakh|lakhs|lac|lacs|m)"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_ocr_confusables(text: str) -> str:
    """Normalize common OCR confusions without touching pure alphabetic words."""
    if not text:
        return ""

    normalized = text
    # Convert common confusions inside tokens containing digits.
    normalized = re.sub(r"(?<=\d)[oO](?=\d)", "0", normalized)
    normalized = re.sub(r"(?<=\d)[iIlL](?=\d)", "1", normalized)
    normalized = re.sub(r"(?<=\d)[sS](?=\d)", "5", normalized)
    normalized = re.sub(r"(?<=\d)[bB](?=\d)", "8", normalized)
    return normalized


def parse_explicit_number(value: str) -> Optional[float]:
    """Parse numbers exactly as present; returns None for invalid values."""
    if value is None:
        return None

    cleaned = value.strip()
    is_parenthesized = cleaned.startswith("(") and cleaned.endswith(")")
    if is_parenthesized:
        cleaned = cleaned[1:-1]

    cleaned = cleaned.replace(",", "")
    if not cleaned:
        return None

    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        return None

    try:
        parsed = float(cleaned)
        if is_parenthesized:
            parsed = -parsed
        return parsed
    except ValueError:
        return None


def _unit_multiplier(unit: Optional[str]) -> float:
    if not unit:
        return 1.0
    return UNIT_MULTIPLIERS_TO_CRORE.get(unit.lower(), 1.0)


def parse_approximate_value(text: str) -> Optional[Dict]:
    """
    Parse approximate/fuzzy numeric values.
    Returns dict with 'value', 'confidence', 'is_approximate', 'has_range'.
    Examples:
    - "~60-65 Cr" → {'value': 62.5, 'confidence': 'low', 'is_approximate': True, 'has_range': True}
    - "around 100 Cr" → {'value': 100.0, 'confidence': 'low', 'is_approximate': True, 'has_range': False}
    - "100 Cr" → {'value': 100.0, 'confidence': 'high', 'is_approximate': False, 'has_range': False}
    """
    if not text or not isinstance(text, str):
        return None

    text = normalize_ocr_confusables(text.strip())

    # Detect approximate indicators
    approx_pattern = r"~|approx(?:imately)?|around|circa|~?(?:about|roughly)"
    has_approximation = bool(re.search(approx_pattern, text, re.IGNORECASE))

    # Try to extract range first (e.g., "60-65", "60 to 65", "2 Cr to 3 Cr")
    number_token = r"(?:\(?[-+]?\d[\d,]*(?:\.\d+)?\)?)"
    range_pattern = (
        rf"({number_token})"
        rf"(?:\s*({UNIT_TOKEN_PATTERN}))?"
        r"\s*(?:[-–—]|to)\s*"
        rf"({number_token})"
        rf"(?:\s*({UNIT_TOKEN_PATTERN}))?"
    )
    range_match = re.search(range_pattern, text)

    if range_match:
        try:
            val1 = parse_explicit_number(range_match.group(1))
            val2 = parse_explicit_number(range_match.group(3))
            if val1 is None or val2 is None:
                return None

            unit1 = range_match.group(2)
            unit2 = range_match.group(4)
            multiplier_1 = _unit_multiplier(unit1 or unit2)
            multiplier_2 = _unit_multiplier(unit2 or unit1)

            normalized_val1 = val1 * multiplier_1
            normalized_val2 = val2 * multiplier_2
            midpoint = (normalized_val1 + normalized_val2) / 2
            range_confidence = 'low' if has_approximation else 'medium'
            return {
                'value': midpoint,
                'confidence': range_confidence,
                'is_approximate': has_approximation,
                'has_range': True,
                'min': min(normalized_val1, normalized_val2),
                'max': max(normalized_val1, normalized_val2),
                'unit': 'crore',
            }
        except (ValueError, AttributeError):
            pass

    # Extract single number
    number_pattern = rf"({number_token})(?:\s*({UNIT_TOKEN_PATTERN}))?"
    number_match = re.search(number_pattern, text)

    if number_match:
        try:
            parsed = parse_explicit_number(number_match.group(1))
            if parsed is None:
                return None

            multiplier = _unit_multiplier(number_match.group(2))
            value = parsed * multiplier
            confidence = 'low' if has_approximation else 'high'
            return {
                'value': value,
                'confidence': confidence,
                'is_approximate': has_approximation,
                'has_range': False,
                'unit': 'crore',
            }
        except ValueError:
            pass

    return None


def sentence_split(text: str) -> list[str]:
    if not text.strip():
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def clean_extracted_text(text: str) -> str:
    """Normalize broken line fragments into cleaner text while preserving values."""
    if not text.strip():
        return ""

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not raw_lines:
        return ""

    merged_lines: list[str] = []
    for line in raw_lines:
        if not merged_lines:
            merged_lines.append(line)
            continue

        previous = merged_lines[-1]

        # Join hard word-breaks from OCR/PDF extraction.
        if previous.endswith("-") and line and line[0].isalnum():
            merged_lines[-1] = previous[:-1] + line
            continue

        # Join obvious broken fragments such as "bles collection.".
        if line and line[0].islower() and not previous.endswith((".", "!", "?", ":", ";")):
            merged_lines[-1] = previous + " " + line
            continue

        merged_lines.append(line)

    return "\n".join(merged_lines)
