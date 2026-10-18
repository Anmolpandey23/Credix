from .models import Signal
from typing import List
from .math_utils import clamp

sector_risk_scores = {
    "construction": 70,
    "real_estate": 65,
    "manufacturing": 55,
    "retail": 50,
    "hospitality": 60,
    "it": 30,
    "healthcare": 40,
    "agriculture": 55,
    "finance": 45,
    "education": 35
}

default_sector_score = 50

amplification_factors = {
    "high_risk": 1.20,
    "medium_risk": 1.10,
    "low_risk": 1.00
}

def compute_sector_score(signals: List[Signal]) -> float:
    sector_signals = [s for s in signals if s.type == "sector"]

    if len(sector_signals) == 0:
        return default_sector_score

    dominant = max(sector_signals, key=lambda s: s.confidence)

    if dominant.evidence is not None:
        sector_name = dominant.evidence.lower().strip()
        if sector_name in sector_risk_scores:
            return sector_risk_scores[sector_name]

    severity_to_score = {
        "low": 25,
        "medium": 50,
        "high": 70,
        "critical": 90
    }
    return severity_to_score.get(dominant.severity.value, default_sector_score)

def apply_sector_amplification(financial_score: float, sector_score: float) -> float:
    if sector_score > 60:
        factor = amplification_factors["high_risk"]
    elif sector_score >= 40:
        factor = amplification_factors["medium_risk"]
    else:
        factor = amplification_factors["low_risk"]

    return clamp(financial_score * factor, 0, 100)
