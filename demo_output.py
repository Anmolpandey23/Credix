"""
Demo script showcasing the improved AIBrain pipeline output.

Demonstrates:
1. Structured warnings instead of raw logs
2. Meaningful financial insights from ratio analysis
3. Clean, deterministic output
4. Confidence-based risk assessment
"""

import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "ai-brain"))

from main import AIBrainPipeline


def demo_clean_output():
    """Demonstrate clean pipeline output with structured warnings and insights."""

    # Create test data
    test_data = {
        "Financial Statements": {
            "Revenue": "1000 Cr",
            "Net Profit": "150 Cr",
            "Total Debt": "800 Cr",
            "Bank Deposits": "700 Cr",
            "GST Revenue": "950 Cr",
            "Receivables": "600 Cr",
            "Operating Cash Flow": "80 Cr",
        }
    }

    # Run pipeline
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_file = tmp_path / "demo_financials.json"

        import json as json_lib
        test_file.write_text(json_lib.dumps(test_data))

        pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
        result = pipeline.run("Demo Company", [test_file])

    return result


def demo_approximate_values():
    """Demonstrate output with approximate and range values."""

    test_data = {
        "Q3 FY2024": [
            "Revenue: 950-1050 Cr",
            "Net Profit: ~140-160 Cr",
            "Total Debt: around 800 Cr",
            "Bank Deposits: 700-750 Cr",
            "GST Revenue: approx 900-1000 Cr",
            "Receivables: ca. 550-600 Cr",
            "Operating Cash Flow: est. 70-90 Cr",
        ]
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_file = tmp_path / "approx_financials.json"

        import json as json_lib
        test_file.write_text(json_lib.dumps(test_data))

        pipeline = AIBrainPipeline(storage_dir=tmp_path / "storage")
        result = pipeline.run("Approximate Company", [test_file])

    return result


if __name__ == "__main__":
    print("=" * 80)
    print("AIBrain Pipeline - Improved Output Demo")
    print("=" * 80)
    print()

    print("1. CLEAN OUTPUT WITH STRUCTURED WARNINGS AND INSIGHTS")
    print("-" * 80)
    result = demo_clean_output()
    print(json.dumps(result, indent=2))

    print()
    print()
    print("2. OUTPUT WITH APPROXIMATE VALUES AND MEANINGFUL WARNINGS")
    print("-" * 80)
    result = demo_approximate_values()
    print(json.dumps(result, indent=2))
