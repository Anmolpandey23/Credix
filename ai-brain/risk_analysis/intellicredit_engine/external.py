from typing import List, Tuple
from .models import Signal
from .math_utils import clamp

severity_scores = {
    "low": 20,
    "medium": 50,
    "high": 75,
    "critical": 95
}

type_fallback = {
    "taxation": "legal",
    "regulatory": "legal",
    "supply_chain": "operational",
    "reputation": "market",
    "credit": "fraud"
}

type_weights = {
    "legal": 0.30,
    "fraud": 0.30,
    "operational": 0.20,
    "market": 0.10,
    "sector": 0.10
}

default_type_weight = 0.10

def compute_external_risk(signals: List[Signal]) -> Tuple[float, List[str]]:
    reasons = []

    if len(signals) == 0:
        return 50.0, ["No external signals available - defaulting to neutral score"]

    grouped = group_signals_by_type(signals)

    total_score = 0.0
    total_weight = 0.0

    for signal_type, type_signals in grouped.items():
        dominant = max(type_signals, key=lambda s: s.confidence)

        base_score = severity_scores.get(dominant.severity.value, 50)

        adjusted_score = base_score * dominant.confidence

        mapped_type = type_fallback.get(signal_type, signal_type)
        weight = type_weights.get(mapped_type, default_type_weight)
        total_score += adjusted_score * weight
        total_weight += weight

        if dominant.severity.value in ["high", "critical"]:
            reasons.append(
                f"{mapped_type.capitalize()} risk signal: {dominant.severity.value} "
                f"( confidence: {dominant.confidence:.0%})"
            )

    if total_weight == 0:
        return 50.0, ["Signal weights summed to zero - defaulting to neutral"]

    final_score = total_score / sum(type_weights.values())

    if len(reasons) == 0:
        reasons.append("No high severity external signals detected")

    return clamp(final_score, 0, 100), reasons

def group_signals_by_type(signals: List[Signal]) -> dict:
    grouped = {}
    for signal in signals:
        if signal.type not in grouped:
            grouped[signal.type] = []
        grouped[signal.type].append(signal)
    return grouped
