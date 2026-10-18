from typing import List

module_priority = {
    "fraud": 1,
    "financial": 2,
    "external": 3,
    "operational": 4,
    "missing": 5
}

high_priority_keywords = [
    "critical", "fraud", "negative", "below", "missing", "concealment", "suspicious", "inflation", "fabrication"
]

def generate_ranked_reasons(financial_reasons: List[str], fraud_reasons: List[str], external_reasons: List[str], operational_reasons: List[str], missing_reasons: List[str]) -> List[str]:
    tagged = []
    tagged.extend(tag_reasons(fraud_reasons, "fraud"))
    tagged.extend(tag_reasons(financial_reasons, "financial"))
    tagged.extend(tag_reasons(external_reasons, "external"))
    tagged.extend(tag_reasons(operational_reasons, "operational"))
    tagged.extend(tag_reasons(missing_reasons, "missing"))

    tagged.sort(key=lambda r: (
        module_priority.get(r["module"], 99),
        0 if contains_high_priority_keyword(r["reason"]) else 1
    ))

    deduplicated = deduplicate_reasons(tagged)
    return [r["reason"] for r in deduplicated]

def tag_reasons(reasons: List[str], module: str) -> List[dict]:
    return [{"reason": r, "module": module} for r in reasons]

def contains_high_priority_keyword(reason: str) -> bool:
    reason_lower = reason.lower()
    return any(keyword in reason_lower for keyword in high_priority_keywords)

def deduplicate_reasons(tagged: List[dict]) -> List[dict]:
    seen = set()
    unique = []
    for item in tagged:
        key = item["reason"][:40].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
