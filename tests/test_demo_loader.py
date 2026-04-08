"""
Tests for demo scenario normalization.
"""

from datetime import date

from evaluator.demo_loader import load_demo_scenarios, resolve_demo_date


class TestDemoScenarioLoading:

    def test_resolve_demo_date_supports_relative_placeholder(self):
        assert resolve_demo_date("__DAYS_AGO:10__", today=date(2026, 4, 7)) == date(2026, 3, 28)

    def test_resolve_demo_date_supports_iso_date(self):
        assert resolve_demo_date("2025-04-01", today=date(2026, 4, 7)) == date(2025, 4, 1)

    def test_load_demo_scenarios_normalizes_chunk_dates_to_iso(self):
        demos = load_demo_scenarios(today=date(2026, 4, 7))
        fed = next(demo for demo in demos if demo["name"] == "🏦 Fed Rate Policy")
        assert fed["chunks"][0]["date"] == "2025-12-18"

    def test_load_demo_scenarios_preserves_raw_date_for_traceability(self):
        demos = load_demo_scenarios(today=date(2026, 4, 7))
        fed = next(demo for demo in demos if demo["name"] == "🏦 Fed Rate Policy")
        assert fed["chunks"][0]["raw_date"] == "__DAYS_AGO:110__"

    def test_load_demo_scenarios_adds_chunk_count_metadata(self):
        demos = load_demo_scenarios(today=date(2026, 4, 7))
        assert all(demo["chunk_count"] == len(demo["chunks"]) for demo in demos)
