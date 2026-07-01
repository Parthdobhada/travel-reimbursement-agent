"""
Reusable helpers for graph nodes and tools.
"""

import json
import re
from typing import Any, Dict, Iterable, List


POLICY_REFERENCE_PATTERN = re.compile(r"POL-[A-Z]+-[A-Z0-9]+")


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from an LLM response.

    Args:
        text: Raw model response.

    Returns:
        Parsed JSON object, or an empty dictionary when parsing fails.
    """

    if not text:
        return {}

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def extract_policy_references(text: str) -> List[str]:
    """
    Extract policy IDs such as POL-HOTEL-001 from text.

    Args:
        text: Policy context or explanation text.

    Returns:
        Stable, de-duplicated list of policy IDs.
    """

    return unique_preserve_order(POLICY_REFERENCE_PATTERN.findall(text or ""))


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    """
    De-duplicate values while preserving first-seen order.

    Args:
        values: Iterable of string values.

    Returns:
        Ordered list with duplicates removed.
    """

    seen = set()
    output = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


def percentage(value: float) -> str:
    """
    Convert a 0-100 score into percentage text.

    Args:
        value: Numeric confidence score.

    Returns:
        Percentage string without decimal noise.
    """

    bounded = max(0.0, min(100.0, value))
    return f"{round(bounded)}%"
