import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ai-brain"))

from cross_check.context import extract_financial_context  # noqa: E402


def test_explicit_fiscal_year_and_declared_unit_are_extracted():
    context = extract_financial_context(
        "Financial Year FY 2025-26. All amounts are stated in INR crore."
    )
    assert context["period"] == "FY2025-2026"
    assert context["currency"] == "INR"
    assert context["unit"] == "crore"
    assert context["basis"] is None


def test_rupee_lakh_declaration_is_extracted():
    context = extract_financial_context(
        "FY 2024-25. Figures in ₹ lakhs."
    )
    assert context["period"] == "FY2024-2025"
    assert context["currency"] == "INR"
    assert context["unit"] == "lakh"


def test_basis_requires_explicit_label_in_field_evidence():
    context = extract_financial_context(
        "FY 2025-26. All amounts in INR crore.",
        "Revenue: 100; basis: audited revenue",
    )
    assert context["basis"] == "audited_revenue"


def test_unlabelled_basis_is_not_inferred():
    context = extract_financial_context(
        "Audited financial statements. FY 2025-26. Amounts in INR crore.",
        "Revenue: 100",
    )
    assert context["basis"] is None


def test_conflicting_periods_remain_unknown():
    context = extract_financial_context(
        "FY 2024-25 and FY 2025-26. Amounts in INR crore."
    )
    assert context["period"] is None


def test_missing_context_remains_unknown():
    context = extract_financial_context(
        "Revenue was 100 and debt was 30.",
        "Revenue: 100",
    )
    assert context == {
        "period": None,
        "currency": None,
        "unit": None,
        "basis": None,
    }


def test_unit_is_not_guessed_from_a_bare_number_suffix():
    context = extract_financial_context(
        "Revenue: 100 crore. FY 2025-26."
    )
    assert context["unit"] is None
