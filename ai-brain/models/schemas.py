"""Typed models for AI Brain pipeline outputs."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal


@dataclass(frozen=True)
class FinancialValue:
    """Encapsulates a financial value with confidence and metadata."""
    value: float
    confidence: Literal['high', 'medium', 'low'] = 'high'
    is_approximate: bool = False
    has_range: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass(frozen=True)
class InsightResult:
    insights: List[str]
    source_chunks: List[str]


@dataclass(frozen=True)
class RuleEvaluation:
    flags: List[str]
    diagnostics: Dict[str, Optional[float]]


# For backward compatibility and simplicity in output JSON, store as float
# but track confidence separately in a parallel dict
Financials = Dict[str, Optional[float]]
FinancialMetadata = Dict[str, Optional[FinancialValue]]
