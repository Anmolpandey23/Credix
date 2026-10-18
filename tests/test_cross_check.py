from cross_check.rule_engine import RuleEngine


def test_bank_deposits_below_gst_revenue_flags_revenue_mismatch():
    financials = {
        "revenue": 100.0,
        "net_profit": 10.0,
        "total_debt": 30.0,
        "bank_deposits": 60.0,
        "gst_revenue": 100.0,
        "receivables": 20.0,
        "operating_cash_flow": 8.0,
    }

    result = RuleEngine().evaluate(financials)

    assert "Revenue mismatch - possible inflation" in result.flags
    assert result.diagnostics["revenue_realization_ratio"] == 0.6


def test_matching_gst_and_financial_revenue_has_no_revenue_difference():
    financials = {
        "revenue": 100.0,
        "net_profit": 10.0,
        "total_debt": 30.0,
        "bank_deposits": 100.0,
        "gst_revenue": 100.0,
        "receivables": 20.0,
        "operating_cash_flow": 8.0,
    }

    result = RuleEngine().evaluate(financials)

    assert (
        "Revenue inconsistency between GST and financials"
        not in result.flags
    )
    assert result.diagnostics[
        "gst_financial_revenue_difference_ratio"
    ] == 0.0 or result.diagnostics[
        "gst_financial_revenue_difference_ratio"
    ] is None


def test_missing_values_do_not_create_division_errors():
    financials = {
        "revenue": None,
        "net_profit": None,
        "total_debt": None,
        "bank_deposits": None,
        "gst_revenue": None,
        "receivables": None,
        "operating_cash_flow": None,
    }

    result = RuleEngine().evaluate(financials)

    assert result.diagnostics["revenue_realization_ratio"] is None
    assert result.diagnostics["profit_cashflow_ratio"] is None
    assert "Insufficient financial data" in result.flags
