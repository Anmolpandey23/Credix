
import math
from typing import List, Tuple

from .math_utils import clamp
from .models import RiskInput


weights = {
    "financial": 0.35,
    "fraud": 0.30,
    "external": 0.20,
    "operational": 0.15,
}

risk_categories = [
    (75, "High Risk"),
    (45, "Medium Risk"),
    (0, "Low Risk"),
]


def _validate_score_inputs(
    financial_score: float,
    fraud_score: float,
    external_score: float,
    operational_score: float,
    missing_penalty: float,
) -> None:
    """Validate module scores and the missing-data penalty."""
    module_scores = {
        "financial_score": financial_score,
        "fraud_score": fraud_score,
        "external_score": external_score,
        "operational_score": operational_score,
    }

    for name, value in module_scores.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{name} must be a finite number between 0 and 100"
            )

        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(
                f"{name} must be a finite number between 0 and 100"
            )

    if (
        isinstance(missing_penalty, bool)
        or not isinstance(missing_penalty, (int, float))
        or not math.isfinite(missing_penalty)
        or missing_penalty < 0
    ):
        raise ValueError(
            "missing_penalty must be a finite, non-negative number"
        )


def _calculate_score_components(
    financial_score: float,
    fraud_score: float,
    external_score: float,
    operational_score: float,
    missing_penalty: float,
) -> Tuple[float, float, float]:
    """Calculate weighted sum, raw score, and clamped final score."""
    weighted_sum = (
        financial_score * weights["financial"]
        + fraud_score * weights["fraud"]
        + external_score * weights["external"]
        + operational_score * weights["operational"]
    )

    raw_score = weighted_sum + missing_penalty
    final_score = clamp(raw_score, 0, 100)

    return weighted_sum, raw_score, final_score


def aggregate_scores(
    financial_score: float,
    fraud_score: float,
    external_score: float,
    operational_score: float,
    missing_penalty: float,
) -> float:
    """Return the validated aggregate risk score."""
    _validate_score_inputs(
        financial_score,
        fraud_score,
        external_score,
        operational_score,
        missing_penalty,
    )

    _, _, final_score = _calculate_score_components(
        financial_score,
        fraud_score,
        external_score,
        operational_score,
        missing_penalty,
    )

    return final_score


def build_score_audit(
    financial_score: float,
    fraud_score: float,
    external_score: float,
    operational_score: float,
    missing_penalty: float,
) -> dict:
    """Return a reproducible audit of the aggregate risk score.

    Weighted contributions are final-score points, not probabilities.
    This records the arithmetic; it does not establish model calibration.
    """
    _validate_score_inputs(
        financial_score,
        fraud_score,
        external_score,
        operational_score,
        missing_penalty,
    )

    module_scores = {
        "financial": float(financial_score),
        "fraud": float(fraud_score),
        "external": float(external_score),
        "operational": float(operational_score),
    }

    # Calculate using full precision.
    weighted_sum, raw_score, final_score = _calculate_score_components(
        financial_score,
        fraud_score,
        external_score,
        operational_score,
        missing_penalty,
    )

    # Round only the values presented in the audit.
    contributions = {
        name: round(score * weights[name], 6)
        for name, score in module_scores.items()
    }

    return {
        "method": "weighted_module_scores_plus_missing_data_penalty",
        "score_direction": "higher_score_means_higher_risk",
        "weights": dict(weights),
        "module_scores": {
            name: round(score, 6)
            for name, score in module_scores.items()
        },
        "weighted_contributions": contributions,
        "weighted_sum": round(float(weighted_sum), 6),
        "missing_data_penalty": round(float(missing_penalty), 6),
        "raw_score_before_clamping": round(float(raw_score), 6),
        "final_score_after_clamping": round(float(final_score), 2),
        "clamped": not 0 <= raw_score <= 100,
    }


def classify_risk(score: float) -> str:
    """Classify a risk score using the existing thresholds."""
    for threshold, category in risk_categories:
        if score >= threshold:
            return category

    return "Low Risk"


def compute_missing_penalty(
    validated: RiskInput,
) -> Tuple[float, List[str]]:
    """Calculate a missing-data penalty and explain missing critical fields."""
    penalty = 0.0
    reasons = []

    critical_fields = [
        (validated.financials.EBIT, "EBIT"),
        (validated.financials.cash_flow, "Cash Flow"),
        (validated.financials.revenue_history, "Revenue History"),
        (validated.financials.gst_revenue, "GST Revenue"),
        (validated.operational_data.workforce, "Workforce"),
        (
            validated.operational_data.capacity_utilization,
            "Capacity Utilization",
        ),
    ]

    missing_count = 0

    for value, name in critical_fields:
        if value is None:
            missing_count += 1
            reasons.append(
                f"{name} missing - incomplete risk profile"
            )

    if missing_count >= 4:
        penalty = 15.0
        reasons.append(
            "Severely incomplete data - elevated assessment uncertainty; "
            "additional financial and operational evidence is required"
        )
    elif missing_count >= 2:
        penalty = 7.0
    elif missing_count == 1:
        penalty = 3.0

    return penalty, reasons
