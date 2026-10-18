"""Validate primary inspection inputs for IntelliCredit."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


_ALLOWED_SEVERITIES = {"low", "medium", "high", "critical"}
_ALLOWED_CATEGORIES = {
    "capacity",
    "machinery",
    "workforce",
    "inventory",
    "safety",
    "compliance",
    "other",
}


def validate_inspection(
    inspection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate inspection evidence without inferring unobserved facts.

    capacity_utilization uses a fraction from 0.0 to 1.0 because that is
    what the current operational scoring engine expects.
    """
    if inspection is None:
        return {
            "status": "not_provided",
            "operational_data": {},
            "findings": [],
            "warnings": ["No primary inspection data was provided."],
        }

    if not isinstance(inspection, Mapping):
        raise ValueError("Inspection data must be a JSON object.")

    operational: dict[str, Any] = {}
    warnings: list[str] = []

    capacity = inspection.get("capacity_utilization")
    if capacity is not None:
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, (int, float))
            or not math.isfinite(float(capacity))
            or not 0 <= float(capacity) <= 1
        ):
            raise ValueError(
                "capacity_utilization must be a number between 0 and 1 "
                "(for example, 0.75 for 75%)."
            )
        operational["capacity_utilization"] = float(capacity)

    machinery = inspection.get("machinery")
    if machinery is not None:
        if not isinstance(machinery, str) or not machinery.strip():
            raise ValueError("machinery must be a non-empty string.")
        operational["machinery"] = machinery.strip()

    for field in ("workforce", "expected_workforce"):
        value = inspection.get(field)
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(f"{field} must be a non-negative integer.")
        operational[field] = value

    raw_findings = inspection.get("findings", [])
    if not isinstance(raw_findings, list):
        raise ValueError("findings must be a list of inspection findings.")

    findings: list[dict[str, Any]] = []
    for index, finding in enumerate(raw_findings):
        if not isinstance(finding, Mapping):
            raise ValueError(f"Finding {index + 1} must be an object.")

        category = finding.get("category", "other")
        severity = finding.get("severity", "medium")
        observation = finding.get("observation")
        evidence = finding.get("evidence")

        if category not in _ALLOWED_CATEGORIES:
            raise ValueError(
                f"Finding {index + 1} has an unsupported category."
            )
        if severity not in _ALLOWED_SEVERITIES:
            raise ValueError(
                f"Finding {index + 1} has an unsupported severity."
            )
        if not isinstance(observation, str) or not observation.strip():
            raise ValueError(
                f"Finding {index + 1} needs an observation."
            )
        if evidence is not None and not isinstance(evidence, str):
            raise ValueError(
                f"Finding {index + 1} evidence must be text."
            )

        findings.append(
            {
                "category": category,
                "severity": severity,
                "observation": observation.strip(),
                "evidence": evidence.strip() if evidence else None,
                "observed_at": finding.get("observed_at"),
                "reviewer": finding.get("reviewer"),
            }
        )

    if not operational and not findings:
        warnings.append(
            "Inspection was supplied but contains no supported indicators "
            "or documented findings."
        )

    return {
        "status": "validated",
        "operational_data": operational,
        "findings": findings,
        "warnings": warnings,
    }
