"""Configuration constants for AI Brain."""

from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".json"}
FINANCIAL_FIELDS = (
    "revenue",
    "net_profit",
    "total_debt",
    "bank_deposits",
    "gst_revenue",
    "receivables",
    "operating_cash_flow",
)

DEFAULT_STORAGE_DIR = Path("./storage")
MAX_INSIGHTS = 8
CHUNK_SIZE = 420
CHUNK_OVERLAP = 80
