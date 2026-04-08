"""
Tests for the Context Freshness Evaluator core modules.

Run with:  pytest tests/ -v

Covers:
  - freshness_score: exponential decay correctness
  - rerank_chunks:   composite ordering + rank assignment
  - get_stale_chunks: staleness threshold filtering
"""

import math
import pytest
from datetime import datetime, timedelta

from evaluator.freshness import freshness_score, days_old
from evaluator.reranker import rerank_chunks, get_stale_chunks


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def make_date(days_ago: int) -> datetime:
    """Return a datetime that is `days_ago` days before now."""
    return datetime.now() - timedelta(days=days_ago)


def make_chunk(relevance: float, days_ago: int, lam: float = 0.03) -> dict:
    """Create a scored chunk dict ready to pass to rerank_chunks."""
    fresh = freshness_score(make_date(days_ago), lam)
    return {
        "text": f"Chunk aged {days_ago}d",
        "date": (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
        "days_old": days_ago,
        "relevance_score": relevance,
        "freshness_score": fresh,
    }


# ══════════════════════════════════════════════════════════════
#  freshness_score tests
# ══════════════════════════════════════════════════════════════

class TestFreshnessScore:

    def test_today_is_perfect(self):
        """A chunk from today should have freshness ≈ 1.0."""
        score = freshness_score(make_date(0))
        assert score == pytest.approx(1.0, abs=0.002), \
            f"Expected ~1.0 for 0-day-old chunk, got {score}"

    def test_decay_is_monotonic(self):
        """Older chunks must always be less fresh than newer ones."""
        scores = [freshness_score(make_date(d)) for d in [0, 7, 30, 90, 365]]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], \
                f"Score did not decay: {scores[i]} vs {scores[i+1]} at index {i}"

    def test_known_decay_value_23_days(self):
        """λ=0.03, 23 days → e^(-0.03*23) ≈ 0.5016."""
        expected = round(math.exp(-0.03 * 23), 4)
        assert freshness_score(make_date(23), 0.03) == pytest.approx(expected, abs=0.005)

    def test_score_bounded_0_to_1(self):
        """Freshness must always be in [0, 1]."""
        for days in [0, 1, 7, 30, 90, 365, 1000]:
            s = freshness_score(make_date(days))
            assert 0.0 <= s <= 1.0, f"Score {s} out of bounds at {days} days"

    def test_higher_lambda_decays_faster(self):
        """Higher λ should produce a lower freshness for the same aged chunk."""
        days = 30
        slow = freshness_score(make_date(days), decay_lambda=0.01)
        fast = freshness_score(make_date(days), decay_lambda=0.05)
        assert fast < slow, \
            f"Expected fast-decay ({fast}) < slow-decay ({slow}) at {days} days"

    def test_days_old_non_negative(self):
        """days_old() should never return a negative value."""
        assert days_old(make_date(0)) >= 0
        assert days_old(make_date(365)) == pytest.approx(365, abs=1)


# ══════════════════════════════════════════════════════════════
#  rerank_chunks tests
# ══════════════════════════════════════════════════════════════

class TestRerankChunks:

    def test_rank_1_has_highest_composite(self):
        """Rank #1 chunk must have the highest composite score."""
        chunks = [make_chunk(0.9, 5), make_chunk(0.4, 300), make_chunk(0.6, 60)]
        ranked = rerank_chunks(chunks)
        assert ranked[0]["rank"] == 1
        composites = [c["composite_score"] for c in ranked]
        assert composites == sorted(composites, reverse=True)

    def test_fresh_relevant_beats_stale_relevant(self):
        """
        KEY PROPERTY: A highly relevant but stale chunk should be demoted
        below a fresh chunk of equal or slightly lower relevance.
        
        This is the core value-prop of the evaluator.
        """
        stale_relevant = make_chunk(relevance=1.0, days_ago=500)   # very stale
        fresh_moderate = make_chunk(relevance=0.8, days_ago=3)      # fresh
        ranked = rerank_chunks([stale_relevant, fresh_moderate])
        # Fresh + moderately relevant should beat stale + fully relevant
        assert ranked[0] is fresh_moderate, (
            f"Expected fresh chunk at rank 1, but stale chunk won. "
            f"Stale composite: {stale_relevant['composite_score']}, "
            f"Fresh composite: {fresh_moderate['composite_score']}"
        )

    def test_ranks_are_1_indexed_and_contiguous(self):
        """Ranks should be 1, 2, 3, … with no gaps."""
        chunks = [make_chunk(r, d) for r, d in [(0.9,5),(0.5,50),(0.2,200)]]
        ranked = rerank_chunks(chunks)
        ranks = sorted(c["rank"] for c in ranked)
        assert ranks == list(range(1, len(chunks) + 1))

    def test_composite_formula_correctness(self):
        """Composite = rel_w * rel + fresh_w * fresh, to 4 decimal places."""
        chunk = make_chunk(relevance=1.0, days_ago=0)  # perfect relevance, perfect freshness
        ranked = rerank_chunks([chunk], relevance_weight=0.6, freshness_weight=0.4)
        expected = round(0.6 * 1.0 + 0.4 * 1.0, 4)
        assert ranked[0]["composite_score"] == pytest.approx(expected, abs=0.001)

    def test_custom_weights_affect_ordering(self):
        """
        With freshness_weight=0.8, a fresh-but-irrelevant chunk should
        beat a stale-but-relevant chunk.
        """
        stale_rel  = make_chunk(relevance=1.0, days_ago=365)
        fresh_irrel = make_chunk(relevance=0.1, days_ago=1)
        ranked = rerank_chunks(
            [stale_rel, fresh_irrel],
            relevance_weight=0.2,
            freshness_weight=0.8
        )
        assert ranked[0] is fresh_irrel, \
            "With high freshness weight, fresh chunk should rank #1"

    def test_single_chunk_gets_rank_1(self):
        """A list with a single chunk should always get rank 1."""
        ranked = rerank_chunks([make_chunk(0.5, 10)])
        assert ranked[0]["rank"] == 1

    def test_empty_list_returns_empty(self):
        """Empty input should return an empty list without errors."""
        assert rerank_chunks([]) == []


# ══════════════════════════════════════════════════════════════
#  get_stale_chunks tests
# ══════════════════════════════════════════════════════════════

class TestGetStaleChunks:

    def test_detects_old_chunk(self):
        """A 3-year-old chunk (λ=0.03) has freshness ≈ 0.0 → must be stale."""
        chunks = rerank_chunks([make_chunk(0.9, 1095)])  # 3 years
        stale = get_stale_chunks(chunks)
        assert len(stale) == 1, "Expected 1 stale chunk but got none"

    def test_fresh_chunk_not_stale(self):
        """A chunk from yesterday is fresh and must not be flagged."""
        chunks = rerank_chunks([make_chunk(0.9, 1)])  # 1 day old
        stale = get_stale_chunks(chunks)
        assert len(stale) == 0, "Yesterday's chunk should not be stale"

    def test_custom_threshold(self):
        """Custom threshold of 0.5 should flag moderately-fresh chunks."""
        chunk = make_chunk(relevance=0.8, days_ago=30)  # ~40% fresh at λ=0.03
        chunks = rerank_chunks([chunk])
        stale = get_stale_chunks(chunks, threshold=0.5)
        assert len(stale) == 1, \
            f"30-day-old chunk (freshness={chunk['freshness_score']}) should be stale at threshold=0.5"

    def test_mixed_batch(self):
        """Only chunks below threshold should be returned."""
        batch = rerank_chunks([
            make_chunk(0.9, 2),    # fresh
            make_chunk(0.5, 500),  # stale
            make_chunk(0.7, 1),    # fresh
            make_chunk(0.3, 800),  # very stale
        ])
        stale = get_stale_chunks(batch, threshold=0.3)
        assert len(stale) == 2, \
            f"Expected 2 stale chunks, got {len(stale)}"
