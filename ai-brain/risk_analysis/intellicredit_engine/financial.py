import math
from typing import List, Tuple, Optional
from .models import Financials
from .math_utils import clamp

# Screening weights sum to 1.0. These are provisional engineering
# defaults, not empirically calibrated lending-model coefficients.
weights = {
    "current_ratio": 0.16,
    "interest_coverage": 0.16,
    "dscr": 0.16,
    "profit_margin": 0.12,
    "cash_flow": 0.12,
    "revenue_trend": 0.08,
    "debt_to_revenue": 0.10,
    "receivables_to_revenue": 0.10,
}

thresholds = {
    "current_ratio_low": 1.0,
    "interest_coverage_low": 1.5,
    "dscr_low": 1.0,
    "profit_margin_low": 0.05,
    "revenue_decline": -0.10,
    "debt_to_revenue_elevated": 0.50,
    "debt_to_revenue_high": 1.00,
    "receivables_to_revenue_elevated": 0.30,
    "receivables_to_revenue_high": 0.50,
}

def compute_financial_risk(financials : Financials) -> Tuple[float, List[str]]:
    score = 0.0
    reasons = []

    # Step 1: Current Ratio = current_assets / current_liabilities
    current_ratio = compute_current_ratio(financials)
    if current_ratio is None:
        score += 50 * weights["current_ratio"]
        reasons.append("Current ratio unavailable - liquidity unknown")
    elif current_ratio < thresholds["current_ratio_low"]:
        score += 80 * weights["current_ratio"]
        reasons.append(f"Low current ratio {current_ratio:.2f} - liquidity risk")
    else:
        score += 20* weights["current_ratio"]

    # Step 2: Interest Coverage Ratio = EBIT / interest_expense
    interest_coverage = compute_interest_coverage(financials)
    if interest_coverage is None:
        score += 50 * weights["interest_coverage"]
        reasons.append("Interest coverage unavailable - debt servicing unknown")
    elif interest_coverage < thresholds["interest_coverage_low"]:
        score += 80 * weights["interest_coverage"]
        reasons.append(f"Low interest coverage {interest_coverage:.2f} - debt risk")
    else:
        score += 20 * weights["interest_coverage"]

    # Step 3: DSCR = net_operating_income / debt/payments
    dscr = compute_dscr(financials)
    if dscr is None:
        score += 50 * weights["dscr"]
        reasons.append("DSCR unavailable - debt coverage unknown")
    elif dscr < thresholds["dscr_low"]:
        score += 90 * weights["dscr"]
        reasons.append(f"DSCR below 1.0 ({dscr:.2f}) - cannot cover debt payments")
    else:
        score += 20 * weights["dscr"]

    # Step 4: Profit Margin = profit / gst_revenue
    profit_margin = compute_profit_margin(financials)
    if profit_margin is None:
        score += 50 * weights["profit_margin"]
        reasons.append("Profit margin unavailable - profitability unknown")
    elif profit_margin < thresholds["profit_margin_low"]:
        score += 75 * weights["profit_margin"]
        reasons.append(f"Low profit margin {profit_margin:.2%} - profitability risk")
    else:
        score += 20 * weights["profit_margin"]

    # Step 5: Cash Flow Check
    if financials.cash_flow is None:
        score += 50 * weights["cash_flow"]
        reasons.append("Cash Flow unavailable - liquidity unknown")
    elif financials.cash_flow < 0:
        score += 85 * weights["cash_flow"]
        reasons.append(f"Negative cash flow {financials.cash_flow:.2f} - critical risk")
    else:
        score += 15 * weights["cash_flow"]

    # Step 6: Revenue Trend = growth rate from revenue_history
    revenue_trend = compute_revenue_trend(financials)
    if revenue_trend is None:
        score += 40 * weights["revenue_trend"]
        reasons.append("Revenue history unavailable - trend unknown")
    elif revenue_trend < thresholds["revenue_decline"]:
        score += 80 * weights["revenue_trend"]
        reasons.append(f"Revenue declining {revenue_trend:.2%} - business deterioration")
    else:
        score += 15 * weights["revenue_trend"]

    # Step 7: Total debt / revenue. Only score with verified comparability.
    debt_ratio = compute_debt_to_revenue(financials)
    if debt_ratio is None:
        score += 50 * weights["debt_to_revenue"]
        reasons.append(
            "Debt-to-revenue ratio unavailable - compatible positive revenue "
            "and verified reporting comparability are required"
        )
    elif debt_ratio > thresholds["debt_to_revenue_high"]:
        score += 80 * weights["debt_to_revenue"]
        reasons.append(
            f"Elevated debt-to-revenue ratio ({debt_ratio:.2f}) - "
            "leverage screening concern"
        )
    elif debt_ratio > thresholds["debt_to_revenue_elevated"]:
        score += 45 * weights["debt_to_revenue"]
        reasons.append(
            f"Moderately elevated debt-to-revenue ratio ({debt_ratio:.2f})"
        )
    else:
        score += 20 * weights["debt_to_revenue"]

    # Step 8: Receivables / revenue. This is a screening indicator, not
    # a substitute for receivables aging, turnover, or collection evidence.
    receivables_ratio = compute_receivables_to_revenue(financials)
    if receivables_ratio is None:
        score += 50 * weights["receivables_to_revenue"]
        reasons.append(
            "Receivables-to-revenue ratio unavailable - compatible positive "
            "revenue and verified reporting comparability are required"
        )
    elif receivables_ratio > thresholds["receivables_to_revenue_high"]:
        score += 75 * weights["receivables_to_revenue"]
        reasons.append(
            f"Elevated receivables-to-revenue ratio ({receivables_ratio:.2f}) - "
            "review collection cycle and receivables aging"
        )
    elif receivables_ratio > thresholds["receivables_to_revenue_elevated"]:
        score += 45 * weights["receivables_to_revenue"]
        reasons.append(
            f"Moderately elevated receivables-to-revenue ratio "
            f"({receivables_ratio:.2f})"
        )
    else:
        score += 20 * weights["receivables_to_revenue"]

    return clamp(score, 0, 100), reasons


def _verified_positive_revenue_ratio(
    numerator: Optional[float],
    revenue: Optional[float],
    comparable: Optional[bool],
) -> Optional[float]:
    """Return a ratio only for finite, valid values with verified context."""
    if comparable is not True:
        return None

    if numerator is None or revenue is None:
        return None

    if (
        isinstance(numerator, bool)
        or isinstance(revenue, bool)
        or not isinstance(numerator, (int, float))
        or not isinstance(revenue, (int, float))
        or not math.isfinite(float(numerator))
        or not math.isfinite(float(revenue))
    ):
        return None

    if numerator < 0 or revenue <= 0:
        return None

    return float(numerator) / float(revenue)


def compute_debt_to_revenue(financials: Financials) -> Optional[float]:
    return _verified_positive_revenue_ratio(
        financials.total_debt,
        financials.revenue,
        financials.debt_revenue_comparable,
    )


def compute_receivables_to_revenue(financials: Financials) -> Optional[float]:
    return _verified_positive_revenue_ratio(
        financials.receivables,
        financials.revenue,
        financials.receivables_revenue_comparable,
    )

# ---- Helper Functions ----

def compute_current_ratio(financials) -> Optional[float]:
    if financials.current_assets is None or financials.current_liabilities is None:
        return None
    if financials.current_liabilities == 0:
        return None
    return financials.current_assets / financials.current_liabilities

def compute_interest_coverage(financials) -> Optional[float]:
    if financials.EBIT is None or financials.interest_expense is None:
        return None
    if financials.interest_expense == 0:
        return None
    return financials.EBIT / financials.interest_expense

def compute_dscr(financials) -> Optional[float]:
    if financials.net_operating_income is None or financials.debt_payments is None:
        return None
    if financials.debt_payments == 0:
        return None
    return financials.net_operating_income / financials.debt_payments

def compute_profit_margin(financials) -> Optional[float]:
    if financials.profit is None or financials.gst_revenue is None:
        return None
    if financials.gst_revenue == 0:
        return None
    return financials.profit / financials.gst_revenue

def compute_revenue_trend(financials) -> Optional[float]:
    history = financials.revenue_history

    if history is None or len(history) < 2:
        return None

    if len(history) == 2:
        first_avg = history[0]
        last_avg = history[1]

    elif len(history) == 3:
        first_avg = history[0]
        last_avg = history[2]

    elif len(history) in [4,5]:
        first_avg = sum(history[0:2])/len(history[0:2])
        last_avg = sum(history[-2:])/len(history[-2:])

    else:
        first_avg = sum(history[0:3])/len(history[0:3])
        last_avg = sum(history[-3:])/len(history[-3:])

    if first_avg == 0:
        return None

    return (last_avg - first_avg) / first_avg
