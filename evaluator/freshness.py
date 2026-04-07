"""
Freshness scoring module.

Uses exponential time decay to score how "fresh" a context chunk is.
Financial data goes stale fast — this module quantifies that decay.

No external dependencies. Pure math, zero latency.
"""

import math
from datetime import datetime


def days_old(chunk_date: datetime) -> int:
    """
    Returns how many days ago chunk_date was from now.
    
    Args:
        chunk_date: The timestamp of the context chunk.
        
    Returns:
        Non-negative integer of days elapsed.
    """
    return max(0, (datetime.now() - chunk_date).days)


def freshness_score(chunk_date: datetime, decay_lambda: float = 0.03) -> float:
    """
    Exponential decay freshness score.
    
    Score = e^(-lambda * days_old)
    
    Examples with lambda=0.03:
        - 0 days old  → 1.0000 (perfectly fresh)
        - 7 days old  → 0.8106
        - 23 days old → 0.5016 (~50%)
        - 100 days old → 0.0498 (~5%)
        
    Financial tuning guide:
        - Price/earnings data: lambda=0.05–0.10 (stale in days)
        - Regulatory/structural: lambda=0.01–0.02 (stale in months)
        - Macro policy data: lambda=0.02–0.04 (stale in weeks)
    
    Args:
        chunk_date: The timestamp of the context chunk.
        decay_lambda: Controls decay speed. Higher = faster staleness.
        
    Returns:
        Float between 0.0 and 1.0 (rounded to 4 decimal places).
    """
    age = days_old(chunk_date)
    return round(math.exp(-decay_lambda * age), 4)
