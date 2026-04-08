"""
Helpers for loading and normalizing demo scenarios.

The demo dataset uses relative date placeholders so the examples stay
meaningful as time passes. This module resolves those placeholders into
stable ISO dates before the UI consumes them.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path


_RELATIVE_DATE_RE = re.compile(r"^__DAYS_AGO:(\d+)__$")


def resolve_demo_date(raw_date: str, today: date | None = None) -> date:
    """
    Convert a demo date string into a concrete date.

    Supported formats:
    - YYYY-MM-DD
    - __DAYS_AGO:<n>__
    """
    base_date = today or date.today()
    match = _RELATIVE_DATE_RE.fullmatch(raw_date.strip())
    if match:
        return base_date - timedelta(days=int(match.group(1)))
    return date.fromisoformat(raw_date)


def load_demo_scenarios(
    source_path: Path | None = None,
    today: date | None = None,
) -> list[dict]:
    """
    Load the demo scenario JSON and normalize chunk dates to ISO strings.
    """
    path = source_path or Path(__file__).parent.parent / "demo_data" / "examples.json"
    scenarios = json.loads(path.read_text())
    normalized = deepcopy(scenarios)

    for scenario in normalized:
        scenario["chunk_count"] = len(scenario.get("chunks", []))
        for chunk in scenario.get("chunks", []):
            resolved_date = resolve_demo_date(chunk["date"], today=today)
            chunk["raw_date"] = chunk["date"]
            chunk["date"] = resolved_date.isoformat()

    return normalized
