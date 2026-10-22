from pydantic import BaseModel , Field
from typing import Optional , List , Literal
from enum import Enum

source_json = "json"
source_web = "web"

class Financials (BaseModel):
    gst_revenue: Optional[float] = None
    bank_deposits: Optional[float] = None
    net_operating_income: Optional[float] = None
    debt_payments: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    EBIT: Optional[float] = None
    interest_expense: Optional[float] = None
    cash_flow: Optional[float] = None
    profit: Optional[float] = None
    revenue_history: Optional[List[float]] = None
    revenue: Optional[float] = None
    total_debt: Optional[float] = None
    receivables: Optional[float] = None
    debt_revenue_comparable: Optional[bool] = None
    receivables_revenue_comparable: Optional[bool] = None
    gst_bank_comparable: Optional[bool] = None
    profit_cashflow_comparable: Optional[bool] = None
    revenue_history_comparable: Optional[bool] = None
    expense_comparable: Optional[bool] = None
    profit_revenue_comparable: Optional[bool] = None

class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Signal (BaseModel):
    type: str
    severity: SeverityLevel
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["json", "web"]
    evidence: Optional[str] = None

class OperationalData (BaseModel):
    capacity_utilization: Optional[float] = None
    machinery: Optional[str] = None
    workforce: Optional[int] = None
    expected_workforce: Optional[int]  = None

class RiskInput (BaseModel):
    company_id: str
    financials: Financials
    external_signals: List[Signal]
    operational_data: OperationalData

class ModuleScores (BaseModel):
    financial: float
    fraud: float
    external: float
    operational: float

class RiskOutput (BaseModel):
    final_score: float
    risk_category: str
    breakdown: ModuleScores
    reasons: List[str]

class MergeResult(BaseModel):
    status: Literal["ok", "error"]
    signals: List[Signal] = []
    warnings: List[str] = []
    message: Optional[str] = None
