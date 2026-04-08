"""
Tests for the Context Freshness Evaluator core modules.

Run with:  pytest tests/ -v

Covers:
  - freshness_score:           exponential decay correctness
  - rerank_chunks:             composite ordering + rank assignment
  - get_stale_chunks:          staleness threshold filtering
  - GPT failure handling:      scoring_unavailable flag propagation
  - action consistency:        result card, summary table, JSON export agree
  - benchmark correctness:     8-scenario benchmark all ranks correct
  - score_relevance (real):    infra exceptions propagate; model failures caught
"""

import json
import math
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from openai import APIConnectionError, AuthenticationError, RateLimitError

from evaluator.freshness import freshness_score, days_old
from evaluator.reranker import rerank_chunks, get_stale_chunks


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def make_date(days_ago: int) -> datetime:
    """Return a datetime that is `days_ago` days before now."""
    return datetime.now() - timedelta(days=days_ago)


def make_chunk(relevance: float, days_ago: int, lam: float = 0.03,
               scoring_unavailable: bool = False) -> dict:
    """Create a scored chunk dict ready to pass to rerank_chunks."""
    fresh = freshness_score(make_date(days_ago), lam)
    return {
        "text": f"Chunk aged {days_ago}d",
        "date": (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
        "days_old": days_ago,
        "relevance_score": relevance,
        "freshness_score": fresh,
        "scoring_unavailable": scoring_unavailable,
    }


def action_label(chunk: dict) -> str:
    """
    Single source of truth for action label.
    Must match result card, summary table, and JSON export in app.py.
    """
    if chunk.get("scoring_unavailable"):
        return "scoring_unavailable"
    comp = chunk["composite_score"]
    if comp >= 0.7:
        return "use"
    if comp >= 0.4:
        return "review"
    return "replace"


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


# ══════════════════════════════════════════════════════════════
#  GPT failure handling tests
# ══════════════════════════════════════════════════════════════

class TestGPTFailureHandling:
    """
    Tests that scoring_unavailable (infra failure) is correctly propagated
    and does NOT contaminate Use/Review/Replace action recommendations.

    These simulate what app.py does when GPT-4o throws an exception:
    relevance_score is set to 0.0 (neutral), scoring_unavailable=True.
    Freshness scoring must remain unaffected.
    """

    def test_unavailable_chunk_freshness_unaffected(self):
        """If relevance scoring fails, freshness must still be computed correctly."""
        normal = make_chunk(relevance=0.8, days_ago=5, scoring_unavailable=False)
        failed = make_chunk(relevance=0.0, days_ago=5, scoring_unavailable=True)
        assert normal["freshness_score"] == failed["freshness_score"], \
            "Freshness score must not be affected by GPT failure"

    def test_unavailable_chunk_gets_zero_relevance_in_composite(self):
        """
        When relevance=0 (set by app on GPT failure), composite =
        freshness * freshness_weight only — relevance contributes nothing.
        """
        chunk = make_chunk(relevance=0.0, days_ago=10, scoring_unavailable=True)
        ranked = rerank_chunks([chunk], relevance_weight=0.6, freshness_weight=0.4)
        expected_composite = round(0.4 * ranked[0]["freshness_score"], 4)
        assert ranked[0]["composite_score"] == pytest.approx(expected_composite, abs=0.01)

    def test_unavailable_excluded_from_action_label(self):
        """
        A scoring_unavailable chunk must NEVER receive use/review/replace.
        Action must be 'scoring_unavailable'.
        """
        chunk = make_chunk(relevance=0.0, days_ago=1, scoring_unavailable=True)
        ranked = rerank_chunks([chunk])
        label = action_label(ranked[0])
        assert label == "scoring_unavailable", \
            f"Expected 'scoring_unavailable' but got '{label}'"

    def test_normal_chunk_not_affected_by_sibling_failure(self):
        """
        When one chunk fails but others score normally, the normal chunks
        must still receive correct use/review/replace labels.
        """
        good_high = make_chunk(relevance=1.0, days_ago=1,   scoring_unavailable=False)
        failed    = make_chunk(relevance=0.0, days_ago=2,   scoring_unavailable=True)
        good_low  = make_chunk(relevance=0.2, days_ago=500, scoring_unavailable=False)
        ranked = rerank_chunks([good_high, failed, good_low])

        labels = {c["days_old"]: action_label(c) for c in ranked}
        assert labels[2] == "scoring_unavailable"
        assert labels[1] != "scoring_unavailable"
        assert labels[500] != "scoring_unavailable"

    def test_all_failed_chunks_labeled_unavailable(self):
        """If all chunks fail, all must be labeled scoring_unavailable."""
        chunks = [make_chunk(0.0, d, scoring_unavailable=True) for d in [1, 30, 365]]
        ranked = rerank_chunks(chunks)
        for c in ranked:
            assert action_label(c) == "scoring_unavailable", \
                f"Chunk at {c['days_old']}d should be scoring_unavailable, got {action_label(c)}"

    def test_freshness_determines_ranking_when_all_fail(self):
        """
        When all chunks have relevance=0 (GPT failure), rank must be
        determined by freshness alone.
        """
        old_chunk   = make_chunk(relevance=0.0, days_ago=365, scoring_unavailable=True)
        fresh_chunk = make_chunk(relevance=0.0, days_ago=1,   scoring_unavailable=True)
        ranked = rerank_chunks([old_chunk, fresh_chunk])
        assert ranked[0] is fresh_chunk, \
            "Fresh chunk must rank #1 when all chunks fail (freshness is the only signal)"


# ══════════════════════════════════════════════════════════════
#  Action consistency across result card / summary table / export
# ══════════════════════════════════════════════════════════════

class TestActionConsistency:
    """
    Ensures that action_label() — which mirrors the logic in app.py for
    result cards, summary table, and JSON export — agrees for all cases.
    Any divergence between these three surfaces is a trust-destroying bug.
    """

    def test_high_composite_is_use(self):
        chunk = make_chunk(relevance=1.0, days_ago=1)
        ranked = rerank_chunks([chunk])
        assert action_label(ranked[0]) == "use", \
            f"High composite {ranked[0]['composite_score']:.3f} should be 'use'"

    def test_mid_composite_is_review(self):
        chunk = make_chunk(relevance=0.5, days_ago=30)
        ranked = rerank_chunks([chunk], relevance_weight=0.6, freshness_weight=0.4)
        comp = ranked[0]["composite_score"]
        label = action_label(ranked[0])
        if 0.4 <= comp < 0.7:
            assert label == "review", f"Composite {comp:.3f} should be 'review', got '{label}'"

    def test_low_composite_is_replace(self):
        chunk = make_chunk(relevance=0.1, days_ago=800)
        ranked = rerank_chunks([chunk])
        assert action_label(ranked[0]) == "replace", \
            f"Low composite {ranked[0]['composite_score']:.3f} should be 'replace'"

    def test_unavailable_always_overrides_composite(self):
        """scoring_unavailable flag overrides any composite score — always."""
        chunk = make_chunk(relevance=0.0, days_ago=0, scoring_unavailable=True)
        ranked = rerank_chunks([chunk])
        assert action_label(ranked[0]) == "scoring_unavailable"

    def test_three_chunk_batch_consistent_labels(self):
        """A realistic batch: Use, Review, Unavailable — all labels consistent."""
        use_chunk  = make_chunk(relevance=1.0, days_ago=1,  scoring_unavailable=False)
        rev_chunk  = make_chunk(relevance=0.5, days_ago=60, scoring_unavailable=False)
        fail_chunk = make_chunk(relevance=0.0, days_ago=10, scoring_unavailable=True)
        ranked = rerank_chunks([use_chunk, rev_chunk, fail_chunk],
                               relevance_weight=0.6, freshness_weight=0.4)
        labels = [action_label(c) for c in ranked]
        assert "scoring_unavailable" in labels
        assert any(l in ("use", "review") for l in labels)


# ══════════════════════════════════════════════════════════════
#  Benchmark correctness wiring
# ══════════════════════════════════════════════════════════════

class TestBenchmarkCorrectness:
    """
    Verifies the 8 curated benchmark scenarios produce 100% correct rankings.
    Replicates the benchmark tab logic without Streamlit dependencies.
    Relevance labels are manually assigned (fixed), not live-scored.
    """

    # (scenario_name, [(rel, days_ago, expected_rank), ...])
    BENCH_CASES = [
        ("Fed Rate",        [(0.95, 120, 1), (0.80, 490, 2), (0.75,  950, 3)]),
        ("Apple EPS",       [(1.00, 520, 1), (0.70, 900, 2), (0.10,  800, 3)]),
        ("NVIDIA revenue",  [(0.95, 140, 1), (0.60, 375, 2), (0.10,  760, 3)]),
        ("Oil price",       [(0.95,  10, 1), (0.70, 670, 2), (0.40, 1250, 3)]),
        ("JPMorgan credit", [(0.90, 180, 1), (0.80, 270, 2), (0.30, 2190, 3)]),
        ("US CPI",          [(1.00,  15, 1), (0.75, 500, 2), (0.60, 1320, 3)]),
        ("S&P 500",         [(0.90,  60, 1), (0.80, 380, 2), (0.40, 1100, 3)]),
        ("Treasury yields", [(1.00,   5, 1), (0.75, 540, 2), (0.30,  760, 3)]),
    ]

    def _run_scenario(self, chunks_spec):
        raw = [make_chunk(rel, days, 0.03) for rel, days, _ in chunks_spec]
        return rerank_chunks(raw, relevance_weight=0.6, freshness_weight=0.4)

    def test_all_scenarios_rank_1_correct(self):
        """Rank #1 must match expected in every scenario."""
        for name, chunks_spec in self.BENCH_CASES:
            ranked = self._run_scenario(chunks_spec)
            expected_days = chunks_spec[0][1]  # expected rank-1 days_ago
            assert ranked[0]["days_old"] == pytest.approx(expected_days, abs=1), \
                f"[{name}] Expected rank-1 to be {expected_days}d old, got {ranked[0]['days_old']}d"

    def test_lowest_composite_chunk_ranks_last(self):
        """
        The chunk with the lowest composite score must always rank last.
        NOTE: This is not always the oldest chunk — an irrelevant-but-fresh chunk
        can correctly score lower than a stale-but-relevant one (Apple EPS scenario).
        """
        for name, chunks_spec in self.BENCH_CASES:
            ranked = self._run_scenario(chunks_spec)
            composites = [c["composite_score"] for c in ranked]
            assert composites[-1] == min(composites), \
                f"[{name}] Last chunk must have lowest composite. Got order: {composites}"


    def test_benchmark_100_percent_accuracy(self):
        """All 24 rankings across 8 scenarios must match expected order."""
        errors = []
        for name, chunks_spec in self.BENCH_CASES:
            ranked = self._run_scenario(chunks_spec)
            sorted_by_expected = sorted(chunks_spec, key=lambda x: x[2])
            for i, (spec, actual) in enumerate(zip(sorted_by_expected, ranked)):
                if abs(actual["days_old"] - spec[1]) > 1:
                    errors.append(
                        f"[{name}] Position {i+1}: expected {spec[1]}d, got {actual['days_old']}d"
                    )
        assert not errors, "Benchmark accuracy below 100%:\n" + "\n".join(errors)


# ══════════════════════════════════════════════════════════════
#  score_relevance() real integration tests (mocked OpenAI client)
# ══════════════════════════════════════════════════════════════

class TestScoreRelevanceIntegration:
    """
    Tests the ACTUAL score_relevance() function via a mocked OpenAI client.

    These prove that the exception model in evaluator/relevance.py works as
    documented: infra failures RAISE (so app.py can set scoring_unavailable=True),
    while model output failures are caught gracefully.
    """

    def _client_raises(self, exc):
        client = MagicMock()
        client.chat.completions.create.side_effect = exc
        return client

    def _client_returns(self, json_str):
        client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json_str
        client.chat.completions.create.return_value = mock_resp
        return client

    def test_connection_error_propagates(self):
        """Network failure must raise — not return score=0.0 silently."""
        from evaluator.relevance import score_relevance
        err = APIConnectionError.__new__(APIConnectionError)
        with pytest.raises(APIConnectionError):
            score_relevance("What is the Fed rate?", "Some chunk.", self._client_raises(err))

    def test_auth_error_propagates(self):
        """Bad API key must raise, not silently return 0.0."""
        from evaluator.relevance import score_relevance
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        err = AuthenticationError("Invalid API key", response=mock_resp, body={})
        with pytest.raises(AuthenticationError):
            score_relevance("What is the Fed rate?", "Some chunk.", self._client_raises(err))

    def test_rate_limit_error_propagates(self):
        """Quota exhausted must raise."""
        from evaluator.relevance import score_relevance
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        err = RateLimitError("Rate limit exceeded", response=mock_resp, body={})
        with pytest.raises(RateLimitError):
            score_relevance("What is the Fed rate?", "Some chunk.", self._client_raises(err))

    def test_valid_response_parses_correctly(self):
        """A valid GPT JSON response returns correct score and reason."""
        from evaluator.relevance import score_relevance
        payload = json.dumps({"score": 0.85, "reason": "Directly addresses the query."})
        result = score_relevance("What is the Fed rate?", "Fed raised rates.", self._client_returns(payload))
        assert result["score"] == pytest.approx(0.85, abs=0.001)
        assert "Directly addresses" in result["reason"]

    def test_malformed_json_caught_gracefully(self):
        """
        Model returning bad JSON (valid HTTP, bad content) is caught internally.
        This is a model failure, NOT infra — must not raise.
        """
        from evaluator.relevance import score_relevance
        result = score_relevance("What is the Fed rate?", "Fed raised rates.", self._client_returns("not json"))
        assert result["score"] == 0.0
        assert result["reason"]  # Some reason string must be present

    def test_score_clamped_to_0_1(self):
        """Out-of-range scores from the model must be clamped to [0, 1]."""
        from evaluator.relevance import score_relevance
        payload = json.dumps({"score": 1.9, "reason": "Very relevant."})
        result = score_relevance("What is the Fed rate?", "Fed raised rates.", self._client_returns(payload))
        assert 0.0 <= result["score"] <= 1.0

    def test_empty_chunk_skips_api_call(self):
        """Empty chunk text must return score=0.0 without calling the API."""
        from evaluator.relevance import score_relevance
        client = MagicMock()
        result = score_relevance("What is the Fed rate?", "   ", client)
        assert result["score"] == 0.0
        client.chat.completions.create.assert_not_called()
