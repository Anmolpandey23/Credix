import logging
from typing import List
from .models import Signal, MergeResult

logger = logging.getLogger(__name__)

def merge_signals(json_signals: List[Signal], web_signals: List[Signal]) -> MergeResult:
    warnings = []

    if len(json_signals) == 0 and len(web_signals) == 0:
        return MergeResult(status= "error", message= "No signals available from any source")

    if len(web_signals) == 0:
        logger.warning("No web signals retrieved - risk assessment based on JSON data only")
        warnings.append("No web signals retrieved - risk assessment based on JSON data only")

    combined = json_signals + web_signals

    deduplicated = deduplicate_signals(combined)
    return MergeResult(status="ok", signals=deduplicated, warnings=warnings)

def deduplicate_signals(signals: List[Signal]) -> List[Signal]:
    seen = {}

    for signal in signals:
        key = (signal.type, signal.source)

        if key not in seen:
            seen[key] = signal
        else:
            if signal.confidence > seen[key].confidence:
                seen[key] = signal
    return list(seen.values())
