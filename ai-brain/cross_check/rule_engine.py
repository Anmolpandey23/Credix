"""Independent deterministic cross-check rules for risk flags."""

from __future__ import annotations

from typing import Dict, Optional

from config.settings import FINANCIAL_FIELDS
from models.schemas import Financials, RuleEvaluation


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


class RuleEngine:
    def evaluate(self, financials: Financials) -> RuleEvaluation:
        flags: list[str] = []
        diagnostics: Dict[str, Optional[float]] = {}

        # Rule 1: Revenue Realization
        ratio_1 = _safe_ratio(financials.get("bank_deposits"), financials.get("gst_revenue"))
        diagnostics["revenue_realization_ratio"] = ratio_1
        if ratio_1 is not None and ratio_1 < 0.75:
            flags.append("Revenue mismatch - possible inflation")

        # Rule 2: Profit vs Cash Flow
        ratio_2 = _safe_ratio(financials.get("operating_cash_flow"), financials.get("net_profit"))
        diagnostics["profit_cashflow_ratio"] = ratio_2
        if ratio_2 is not None and ratio_2 < 0.7:
            flags.append("Profit not backed by cash flow")

        # Rule 3: Receivables Risk
        ratio_3 = _safe_ratio(financials.get("receivables"), financials.get("revenue"))
        diagnostics["receivables_ratio"] = ratio_3
        if ratio_3 is not None and ratio_3 > 0.5:
            flags.append("High receivables risk")

        # Rule 4: Debt Stress
        ratio_4 = _safe_ratio(financials.get("total_debt"), financials.get("revenue"))
        diagnostics["debt_ratio"] = ratio_4
        if ratio_4 is not None and ratio_4 > 0.7:
            flags.append("High leverage risk")

        # Rule 5: GST vs Financial Revenue
        revenue = financials.get("revenue")
        gst_revenue = financials.get("gst_revenue")
        difference_ratio: Optional[float] = None
        if revenue is not None and gst_revenue is not None and revenue != 0 and gst_revenue != revenue:
            difference_ratio = abs(gst_revenue - revenue) / revenue
            if difference_ratio > 0.2:
                flags.append("Revenue inconsistency between GST and financials")
        diagnostics["gst_financial_revenue_difference_ratio"] = difference_ratio

        # Rule 6: Missing Critical Data
        null_count = sum(1 for field in FINANCIAL_FIELDS if financials.get(field) is None)
        if null_count > len(FINANCIAL_FIELDS) / 2:
            flags.append("Insufficient financial data")

        deduped_flags = list(dict.fromkeys(flags))
        return RuleEvaluation(flags=deduped_flags, diagnostics=diagnostics)
