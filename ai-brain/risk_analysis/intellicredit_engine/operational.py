from typing import List, Tuple
from .models import OperationalData
from .math_utils import clamp

weights = {
    "workforce_gap": 0.40,
    "capacity_utilization": 0.40,
    "machinery": 0.20
}

thresholds = {
    "workforce_gap_high": 0.40,
    "capacity_low": 0.50,
    "capacity_high": 0.95
}

def compute_operational_risk(operational : OperationalData) -> Tuple[float, List[str]]:
    score = 0.0
    reasons = []

    if operational.workforce is None or operational.expected_workforce is None:
        score += 50 * weights["workforce_gap"]
        reasons.append("Workforce data unavailable - operational capacity unknown")
    elif operational.expected_workforce == 0:
        score += 50 * weights["workforce_gap"]
        reasons.append("Expected workforce is zero - data error")
    elif operational.workforce > operational.expected_workforce:
        gap = abs(operational.workforce - operational.expected_workforce) / operational.expected_workforce
        if gap > thresholds["workforce_gap_high"]:
            score += 90 * weights["workforce_gap"]
            reasons.append(f"Workforce exceeds expected by {gap:.2%} - possible inflation for loan fraud")
        else:
            score += 15 * weights["workforce_gap"]
    else:
        gap = abs(operational.expected_workforce - operational.workforce) / operational.expected_workforce
        if gap > thresholds["workforce_gap_high"]:
             score += 75 * weights["workforce_gap"]
             reasons.append(f"Workforce below expected by {gap:.2%} - understaffing risk")
        else:
            score += 15 * weights["workforce_gap"]


    if operational.capacity_utilization is None:
        score += 50 * weights["capacity_utilization"]
        reasons.append("Capacity utilization unavailable - efficiency unknown")
    elif operational.capacity_utilization < thresholds["capacity_low"]:
        score += 70 * weights["capacity_utilization"]
        reasons.append(f"Low capacity utilization {operational.capacity_utilization:.0%} - underperforming")
    elif operational.capacity_utilization > thresholds["capacity_high"]:
        score += 65 * weights["capacity_utilization"]
        reasons.append(f"Capacity utilization critical {operational.capacity_utilization:.0%} - overload risk")
    else:
        score += 15 * weights["capacity_utilization"]


    if operational.machinery is None or operational.machinery.strip() == "":
        score += 60 * weights["machinery"]
        reasons.append("Machinery data missing - asset verification impossible")
    else:
        score += 10 * weights["machinery"]

    return clamp(score, 0, 100), reasons
