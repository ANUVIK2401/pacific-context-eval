"""
Reranker module.

Computes composite scores from relevance + freshness signals
and returns chunks sorted by composite rank.
"""


def rerank_chunks(
    chunks: list[dict],
    relevance_weight: float = 0.6,
    freshness_weight: float = 0.4
) -> list[dict]:
    """
    Compute composite score and sort chunks by rank.
    
    Composite = (relevance_weight * relevance) + (freshness_weight * freshness)
    
    Args:
        chunks: List of dicts, each with keys:
            - text (str): Chunk content
            - date (str): ISO date string
            - relevance_score (float): 0-1 relevance score
            - freshness_score (float): 0-1 freshness score
        relevance_weight: Weight for relevance signal (default 0.6)
        freshness_weight: Weight for freshness signal (default 0.4)
    
    Returns:
        Same list with added keys:
            - composite_score (float)
            - rank (int, 1-indexed)
        Sorted descending by composite_score.
    """
    for chunk in chunks:
        chunk["composite_score"] = round(
            relevance_weight * chunk.get("relevance_score", 0.0) +
            freshness_weight * chunk.get("freshness_score", 0.0),
            4
        )

    ranked = sorted(chunks, key=lambda x: x["composite_score"], reverse=True)
    for i, chunk in enumerate(ranked):
        chunk["rank"] = i + 1

    return ranked


def get_stale_chunks(chunks: list[dict], threshold: float = 0.3) -> list[dict]:
    """
    Returns chunks where freshness_score is below threshold.
    Useful for generating stale-data warnings.
    """
    return [c for c in chunks if c.get("freshness_score", 0.0) < threshold]
