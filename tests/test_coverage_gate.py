import json

import pytest

from open_standard_evaluation.models.eval_item import EvalItem
from open_standard_evaluation.models.pattern import PatternEntry, PatternIndex
from open_standard_evaluation.steps.coverage_gate import (
    _is_duplicate_query,
    _format_patterns_for_prompt,
    _heuristic_gate,
    load_pattern_index,
    save_pattern_index,
)


class TestPatternIndex:
    def test_empty_index(self):
        idx = PatternIndex()
        assert idx.total_evals == 0
        assert idx.patterns == []

    def test_add_pattern(self):
        idx = PatternIndex()
        idx.add_pattern(PatternEntry(
            pattern="account lookup",
            description="User asks about an account",
            example="What's happening with Acme?",
            has_eval=True,
            evals_generated=1,
            eval_queries=["What's happening with Acme?"],
        ))
        assert len(idx.patterns) == 1
        assert idx.total_evals == 1

    def test_at_cap(self):
        entry = PatternEntry(
            pattern="test",
            description="test",
            example="test",
            evals_generated=3,
        )
        assert entry.at_cap is True

        entry2 = PatternEntry(
            pattern="test",
            description="test",
            example="test",
            evals_generated=2,
        )
        assert entry2.at_cap is False

    def test_find_match_exact(self):
        idx = PatternIndex(patterns=[
            PatternEntry(pattern="account lookup", description="x", example="x"),
        ])
        assert idx.find_match("account lookup") is not None
        assert idx.find_match("Account Lookup") is not None

    def test_find_match_substring(self):
        idx = PatternIndex(patterns=[
            PatternEntry(pattern="account lookup", description="x", example="x"),
        ])
        assert idx.find_match("account") is not None

    def test_find_match_capability(self):
        idx = PatternIndex(patterns=[
            PatternEntry(
                pattern="data export",
                description="x",
                example="x",
                capabilities=["csv export"],
            ),
        ])
        assert idx.find_match("csv export") is not None

    def test_find_match_no_match(self):
        idx = PatternIndex(patterns=[
            PatternEntry(pattern="account lookup", description="x", example="x"),
        ])
        assert idx.find_match("billing configuration") is None

    def test_get_uncapped_capped(self):
        idx = PatternIndex(patterns=[
            PatternEntry(pattern="a", description="x", example="x", evals_generated=3),
            PatternEntry(pattern="b", description="x", example="x", evals_generated=1),
            PatternEntry(pattern="c", description="x", example="x", evals_generated=0),
        ])
        assert len(idx.get_capped_patterns()) == 1
        assert len(idx.get_uncapped_patterns()) == 2


class TestDuplicateQuery:
    def test_exact_match(self):
        assert _is_duplicate_query("How do I reset my password?", ["How do I reset my password?"])

    def test_high_overlap(self):
        assert _is_duplicate_query(
            "How do I reset my account password?",
            ["How do I reset my password?"],
        )

    def test_no_overlap(self):
        assert not _is_duplicate_query(
            "What's the billing cycle for enterprise plans?",
            ["How do I reset my password?"],
        )

    def test_empty_existing(self):
        assert not _is_duplicate_query("test query", [])


class TestHeuristicGate:
    def test_new_pattern_passes(self):
        candidate = EvalItem(
            user_message="How do I export data?",
            reference_response="Go to...",
            judge_remarks="Tests export",
            cluster_id=1,
            cluster_title="Data Export",
            source_session_id="s1",
            synthesis_type="verbatim",
            capability_tested="data export",
        )
        idx = PatternIndex()
        passes, matched, suggested, cap = _heuristic_gate(candidate, idx)
        assert passes is True
        assert matched is None

    def test_capped_pattern_fails(self):
        candidate = EvalItem(
            user_message="How do I export data?",
            reference_response="Go to...",
            judge_remarks="Tests export",
            cluster_id=1,
            cluster_title="Data Export",
            source_session_id="s1",
            synthesis_type="verbatim",
            capability_tested="data export",
        )
        idx = PatternIndex(patterns=[
            PatternEntry(
                pattern="data export",
                description="x",
                example="x",
                evals_generated=3,
            ),
        ])
        passes, matched, suggested, cap = _heuristic_gate(candidate, idx)
        assert passes is False


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        idx = PatternIndex(patterns=[
            PatternEntry(
                pattern="test",
                description="testing",
                example="test query",
                has_eval=True,
                evals_generated=2,
                eval_queries=["q1", "q2"],
            ),
        ])
        path = tmp_path / "patterns.json"
        save_pattern_index(idx, path)

        loaded = load_pattern_index(path)
        assert len(loaded.patterns) == 1
        assert loaded.patterns[0].pattern == "test"
        assert loaded.patterns[0].evals_generated == 2

    def test_load_nonexistent(self, tmp_path):
        loaded = load_pattern_index(tmp_path / "missing.json")
        assert loaded.patterns == []


class TestFormatPatternsForPrompt:
    def test_empty(self):
        idx = PatternIndex()
        assert "empty" in _format_patterns_for_prompt(idx)

    def test_formats_with_cap_status(self):
        idx = PatternIndex(patterns=[
            PatternEntry(
                pattern="account lookup",
                description="User checks account status",
                example="What about Acme?",
                evals_generated=3,
                eval_queries=["What about Acme?", "Status of BigCorp?"],
            ),
            PatternEntry(
                pattern="billing",
                description="Billing questions",
                example="How much?",
                evals_generated=1,
                eval_queries=["How much?"],
            ),
        ])
        text = _format_patterns_for_prompt(idx)
        assert "AT CAP" in text
        assert "1/3" in text
        assert "account lookup" in text
