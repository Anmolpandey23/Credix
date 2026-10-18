import logging
logger = logging.getLogger(__name__)
from pydantic import BaseModel , ValidationError
from .models import Financials , OperationalData, Signal, RiskInput
from typing import Literal , Optional , List

class ValidationResult (BaseModel):
    status: Literal["ok", "error"]
    data: Optional[RiskInput] = None
    message: Optional[str] = None
    warnings: List[str] = []

def validate_and_standardize(raw_input: dict) -> ValidationResult:
    if raw_input is None or not isinstance(raw_input, dict):
        return ValidationResult(status = "error", message = "Input must be a non-null JSON object")

    try:
        financials = _parse_financials(raw_input.get("financials", {}))
        operational = _parse_operational(raw_input.get("operational_data", {}))
        signals = _parse_signals(raw_input.get("external_signals", []))
        company_id = raw_input.get("company_id")

        if company_id is None or company_id == "":
            return ValidationResult(status = "error", message = "company_id is required and cannot be empty")

        risk_input = RiskInput(
            company_id= company_id,
            financials= financials,
            external_signals= signals,
            operational_data= operational
        )

        quality_issues = _check_data_quality(risk_input)

        if quality_issues:
            logger.warning(quality_issues)
        return ValidationResult(status = "ok", data = risk_input, warnings=quality_issues)

    except ValidationError as e:
        return ValidationResult(status="error", message=f"Schema Validation failed: {e}")

    except KeyError as e:
        return ValidationResult(status="error", message=f"Missing required field: {e}")

    except Exception as e:
        return ValidationResult(status="error", message=f"Unexpected error during validation: {e}")

def _parse_financials(fin_data: dict) -> Financials:
    return Financials(
        gst_revenue=fin_data.get("gst_revenue"),
        bank_deposits=fin_data.get("bank_deposits"),
        net_operating_income=fin_data.get("net_operating_income"),
        debt_payments=fin_data.get("debt_payments"),
        current_assets=fin_data.get("current_assets"),
        current_liabilities=fin_data.get("current_liabilities"),
        EBIT=fin_data.get("EBIT"),
        interest_expense=fin_data.get("interest_expense"),
        cash_flow=fin_data.get("cash_flow"),
        profit=fin_data.get("profit"),
        revenue_history=fin_data.get("revenue_history"),
        revenue=fin_data.get("revenue"),
        total_debt=fin_data.get("total_debt"),
        receivables=fin_data.get("receivables"),
        debt_revenue_comparable=fin_data.get("debt_revenue_comparable"),
        receivables_revenue_comparable=fin_data.get("receivables_revenue_comparable"),
        gst_bank_comparable=fin_data.get("gst_bank_comparable"),
        profit_cashflow_comparable=fin_data.get("profit_cashflow_comparable"),
        revenue_history_comparable=fin_data.get("revenue_history_comparable"),
        expense_comparable=fin_data.get("expense_comparable"),
    )

def _parse_operational(op_data: dict) -> OperationalData:
    return OperationalData(
        capacity_utilization=op_data.get("capacity_utilization"),
        machinery=op_data.get("machinery"),
        workforce=op_data.get("workforce"),
        expected_workforce=op_data.get("expected_workforce")
    )

def _parse_signals(raw_signals: list) -> List[Signal]:
    signals = []
    for i, raw_signal in enumerate(raw_signals):
        try:
            signals.append(Signal(
                type=raw_signal["type"],
                severity=raw_signal["severity"],
                confidence=raw_signal["confidence"],
                source="json"
            ))
        except (KeyError, ValidationError) as e:
            logger.warning(f"Skipping malformed signal at index {i}: {e}")
    return signals

def _check_data_quality(risk_input: RiskInput) -> List[str]:
    issues = []

    if risk_input.financials.EBIT is None and risk_input.financials.interest_expense is None:
        issues.append("EBIT and interest_expense both missing - interest coverage ratio unavailable")

    if risk_input.financials.current_assets is None and risk_input.financials.current_liabilities is None:
        issues.append("current_assets and current_liabilities both missing - liquidity ratio unavailable")

    if risk_input.financials.debt_payments is not None and risk_input.financials.debt_payments < 0:
        issues.append("debt_payments is negative - likely data entry error")

    if risk_input.operational_data.workforce is not None and risk_input.operational_data.workforce < 0:
        issues.append("workforce is negative - invalid value")

    if all(signal.confidence == 0.0 for signal in risk_input.external_signals):
        issues.append("All signals have zero confidence - external data may be unreliable")

    return issues
