"""Financial insight generator for meaningful interpretations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class FinancialInsightGenerator:
    """Generate deterministic financial insights from extracted data and ratios."""

    def __init__(self):
        # Mapping of ratio thresholds to insight messages with severity scoring
        # Severity: 3=critical, 2=high, 1=medium
        self.insight_templates = {
            "weak_cash_flow": {
                "rule": lambda r: r is not None and r < 0.3,
                "insight": "Operating cash flow is critically low relative to net profit (less than 30%), which may suggest significant earnings quality concerns or working capital constraints.",
                "severity": 3,
            },
            "profit_cashflow": {
                "rule": lambda r: r is not None and r < 0.7,
                "insight": "Operating cash flow is substantially lower than net profit, which may indicate potential accounting adjustments and could suggest liquidity pressure.",
                "severity": 3,
            },
            "revenue_realization": {
                "rule": lambda r: r is not None and r < 0.75,
                "insight": "GST revenue significantly exceeds bank deposits, which may suggest extended payment cycles or potential working capital strain.",
                "severity": 2,
            },
            "high_leverage": {
                "rule": lambda r: r is not None and r > 0.7,
                "insight": "Total debt exceeds 70% of revenue, which may constrain borrowing capacity and increase refinancing risk.",
                "severity": 2,
            },
            "high_receivables": {
                "rule": lambda r: r is not None and r > 0.5,
                "insight": "Trade receivables exceed 50% of revenue, which may indicate extended credit terms or collection timing issues.",
                "severity": 1,
            },
        }

    def generate(
        self,
        financials: Dict[str, Optional[float]],
        diagnostics: Dict[str, Optional[float]],
        extraction_metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate ranked financial insights based on extracted data and ratios.

        Returns top 3-4 insights ranked by severity (critical first).
        """
        insights_with_severity: List[tuple[str, int]] = []

        # Extract key ratios from diagnostics
        revenue_realization = diagnostics.get("revenue_realization_ratio")
        profit_cashflow = diagnostics.get("profit_cashflow_ratio")
        receivables_ratio = diagnostics.get("receivables_ratio")
        debt_ratio = diagnostics.get("debt_ratio")

        # Evaluate all templates and collect matching insights with severity
        # Check critical weak_cash_flow first (most severe)
        if profit_cashflow is not None and profit_cashflow < 0.3:
            insights_with_severity.append(
                (self.insight_templates["weak_cash_flow"]["insight"], 3)
            )
        # Otherwise check profit_cashflow (overlaps with weak_cash_flow)
        elif self.insight_templates["profit_cashflow"]["rule"](profit_cashflow):
            insights_with_severity.append(
                (self.insight_templates["profit_cashflow"]["insight"], 3)
            )

        if self.insight_templates["revenue_realization"]["rule"](revenue_realization):
            insights_with_severity.append(
                (self.insight_templates["revenue_realization"]["insight"], 2)
            )

        if self.insight_templates["high_leverage"]["rule"](debt_ratio):
            insights_with_severity.append(
                (self.insight_templates["high_leverage"]["insight"], 2)
            )

        if self.insight_templates["high_receivables"]["rule"](receivables_ratio):
            insights_with_severity.append(
                (self.insight_templates["high_receivables"]["insight"], 1)
            )

        # Sort by severity (descending) to show critical insights first
        insights_with_severity.sort(key=lambda x: x[1], reverse=True)

        # Extract just the insight text, limited to top 3-4
        ranked_insights = [insight for insight, _ in insights_with_severity][:4]

        return ranked_insights

    def generate_warnings(
        self,
        financials: Dict[str, Optional[float]],
        metadata: Optional[Dict[str, Any]] = None,
        confidence_metrics: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate concise, non-redundant warnings about data quality.

        Returns:
            List of warning messages (no duplicates)
        """
        warnings: List[str] = []

        if not confidence_metrics:
            return warnings

        quality_artifacts = confidence_metrics.get("quality_artifacts", {})

        # Consolidate data quality warnings
        approx_fields = quality_artifacts.get("approx_fields", [])
        range_fields = quality_artifacts.get("range_fields", [])

        # Single consolidated warning for approximations/ranges
        if approx_fields or range_fields:
            affected_fields = len(approx_fields) + len(range_fields)
            if affected_fields > 0:
                warnings.append(
                    f"Data quality: {affected_fields} field(s) extracted as approximate or range values. "
                    "Reported values are estimates."
                )

        # Critical field approximation warning (only if NOT already covered above)
        if quality_artifacts.get("critical_approx") and not warnings:
            warnings.append(
                "Critical fields extracted with approximations. Analysis confidence is reduced."
            )

        # High approximate ratio warning (consolidated, not redundant)
        if quality_artifacts.get("high_approximate_ratio") and len(warnings) < 2:
            warnings.append(
                "Over 50% of financial data contains approximations or ranges. "
                "Results should be treated as indicative only."
            )

        # Missing critical fields warning
        missing_critical = []
        for field in ("gst_revenue", "bank_deposits", "revenue"):
            if field not in financials or financials[field] is None:
                missing_critical.append(field)

        if missing_critical and len(warnings) < 3:
            warnings.append(
                f"Missing critical field(s): {', '.join(missing_critical)}. Analysis scope is limited."
            )

        # Limit to 3 most important warnings
        return warnings[:3]
