from typing import List, Optional, Tuple

from .math_utils import clamp
from .models import Financials


weights = {
    "gst_vs_bank": 0.30,
    "profit_vs_cashflow": 0.25,
    "revenue_spike": 0.25,
    "expense_anomaly": 0.20,
}

thresholds = {
    "gst_bank_ratio_low": 0.5,
    "gst_bank_ratio_high": 2.0,
    "profit_cashflow_gap": 0.30,
    "revenue_spike": 0.50,
}


def compute_fraud_risk(financials: Financials) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    evaluated: List[Tuple[float, float]] = []

    def record_signal(name: str, signal_score: float) -> None:
        evaluated.append((weights[name], signal_score))

    # GST revenue and bank deposits must be explicitly comparable.
    if (
        financials.gst_bank_comparable is not True
        or financials.gst_revenue is None
        or financials.bank_deposits is None
    ):
        reasons.append(
            "GST-to-bank comparison not scored: values or verified "
            "comparability context are unavailable"
        )
    elif financials.bank_deposits <= 0:
        reasons.append(
            "GST-to-bank comparison not scored: bank deposits must be "
            "positive; verify statement coverage and reporting context"
        )
    else:
        ratio = financials.gst_revenue / financials.bank_deposits

        if ratio < thresholds["gst_bank_ratio_low"]:
            record_signal("gst_vs_bank", 85)
            reasons.append(
                f"GST revenue is materially lower than reported bank "
                f"deposits (ratio {ratio:.2f}); investigate reporting "
                "periods, statement coverage, and transaction classification"
            )
        elif ratio > thresholds["gst_bank_ratio_high"]:
            record_signal("gst_vs_bank", 85)
            reasons.append(
                f"GST revenue is materially higher than reported bank "
                f"deposits (ratio {ratio:.2f}); investigate reporting "
                "periods, statement coverage, and transaction classification"
            )
        else:
            record_signal("gst_vs_bank", 10)

    # Compare profit and cash flow only when their contexts are verified.
    if (
        financials.profit_cashflow_comparable is not True
        or financials.profit is None
        or financials.cash_flow is None
    ):
        reasons.append(
            "Profit-to-cash-flow comparison not scored: values or verified "
            "comparability context are unavailable"
        )
    else:
        if financials.profit == 0:
            gap = 0.0 if financials.cash_flow == 0 else 1.0
        else:
            gap = abs(
                financials.profit - financials.cash_flow
            ) / abs(financials.profit)

        if gap > thresholds["profit_cashflow_gap"]:
            record_signal("profit_vs_cashflow", 80)
            reasons.append(
                f"Large profit-cash-flow gap ({gap:.2%}); investigate "
                "earnings-to-cash conversion and supporting statements"
            )
        else:
            record_signal("profit_vs_cashflow", 10)

    # Revenue history needs an explicitly comparable sequence.
    if (
        financials.revenue_history_comparable is not True
        or detect_revenue_spike(financials) is None
    ):
        reasons.append(
            "Revenue-spike signal unavailable: comparable revenue history "
            "has not been verified"
        )
    else:
        spike = detect_revenue_spike(financials)
        if spike is not None and spike > thresholds["revenue_spike"]:
            record_signal("revenue_spike", 85)
            reasons.append(
                f"Sudden revenue increase of {spike:.2%}; verify source "
                "statements, reporting periods, and possible one-off events"
            )
        else:
            record_signal("revenue_spike", 10)

    # This is a debt-payment burden indicator, not proof of fraud.
    if (
        financials.expense_comparable is not True
        or financials.net_operating_income is None
        or financials.debt_payments is None
        or financials.net_operating_income <= 0
    ):
        reasons.append(
            "Debt-payment burden signal unavailable: values or verified "
            "comparability context are missing, or operating income is "
            "not positive"
        )
    elif (
        financials.debt_payments
        > financials.net_operating_income * 1.5
    ):
        record_signal("expense_anomaly", 75)
        reasons.append(
            "Debt payments materially exceed operating income; investigate "
            "debt-service capacity and the supporting financial statements"
        )
    else:
        record_signal("expense_anomaly", 10)

    if not evaluated:
        reasons.append(
            "Fraud-risk score is neutral because no signal had sufficient "
            "verified evidence; this does not establish that fraud risk is low"
        )
        return 50.0, reasons

    total_weight = sum(weight for weight, _ in evaluated)
    score = sum(weight * value for weight, value in evaluated) / total_weight

    return clamp(score, 0, 100), reasons


def detect_revenue_spike(financials: Financials) -> Optional[float]:
    history = financials.revenue_history

    if history is None or len(history) < 2:
        return None

    max_spike = 0.0

    for previous, current in zip(history, history[1:]):
        if previous <= 0:
            continue

        jump = (current - previous) / previous
        max_spike = max(max_spike, jump)

    return max_spike
