"""Tests for the output module — jsonl, csv, and braintrust formats."""

import csv
import json
from pathlib import Path

import pytest

from open_standard_evaluation.models.eval_item import EvalItem
from open_standard_evaluation.output.braintrust import write_braintrust
from open_standard_evaluation.output.csv import write_csv
from open_standard_evaluation.output.jsonl import write_jsonl


def _make_eval(**overrides) -> EvalItem:
    defaults = dict(
        user_message="How do I reset my password?",
        reference_response="Go to Settings > Security > Reset Password.",
        judge_remarks="Tests account recovery workflow guidance",
        cluster_id=1,
        cluster_title="Account Management",
        source_session_id="sess_001",
        synthesis_type="verbatim",
        capability_tested="account recovery",
    )
    defaults.update(overrides)
    return EvalItem(**defaults)


class TestJsonlOutput:
    def test_writes_correct_schema(self, tmp_path):
        """jsonl output must use input/ideal/metadata keys — not user_message/reference_response."""
        path = tmp_path / "evals.jsonl"
        write_jsonl([_make_eval()], path)

        row = json.loads(path.read_text().strip())
        assert "input" in row, "jsonl must use 'input' key"
        assert "ideal" in row, "jsonl must use 'ideal' key"
        assert "metadata" in row
        assert row["metadata"]["cluster_id"] == 1
        assert row["metadata"]["synthesis_type"] == "verbatim"

    def test_one_line_per_eval(self, tmp_path):
        evals = [_make_eval(user_message=f"Question {i}") for i in range(5)]
        path = tmp_path / "evals.jsonl"
        write_jsonl(evals, path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 5
        for i, line in enumerate(lines):
            assert json.loads(line)["input"] == f"Question {i}"

    def test_unicode_content(self, tmp_path):
        """Non-ASCII content must survive json round-trip without mojibake."""
        eval_item = _make_eval(
            user_message="日本語の質問です",
            reference_response="日本語の回答です",
        )
        path = tmp_path / "evals.jsonl"
        write_jsonl([eval_item], path)

        row = json.loads(path.read_text().strip())
        assert row["input"] == "日本語の質問です"


class TestBraintrustOutput:
    def test_uses_expected_not_ideal(self, tmp_path):
        """Braintrust format uses 'expected' not 'ideal' — mixing them up breaks eval ingestion."""
        path = tmp_path / "evals.jsonl"
        write_braintrust([_make_eval()], path)

        row = json.loads(path.read_text().strip())
        assert "expected" in row, "braintrust must use 'expected' key"
        assert "ideal" not in row, "braintrust must NOT use 'ideal' key"
        assert row["input"] == "How do I reset my password?"

    def test_metadata_includes_cluster_info(self, tmp_path):
        path = tmp_path / "evals.jsonl"
        write_braintrust([_make_eval()], path)

        row = json.loads(path.read_text().strip())
        assert row["metadata"]["cluster_id"] == 1
        assert row["metadata"]["synthesis_type"] == "verbatim"


class TestCsvOutput:
    def test_csv_headers_match_expected(self, tmp_path):
        """CSV must use user_message/reference_output/remarks — consumed by downstream tools."""
        path = tmp_path / "evals.csv"
        write_csv([_make_eval()], path)

        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert rows[0] == ["user_message", "reference_output", "remarks"]

    def test_csv_handles_commas_in_content(self, tmp_path):
        """Real eval content has commas. If quoting breaks, rows get split wrong."""
        eval_item = _make_eval(
            user_message="A, B, and C",
            reference_response="Step 1, Step 2, Step 3",
        )
        path = tmp_path / "evals.csv"
        write_csv([eval_item], path)

        with open(path) as f:
            rows = list(csv.reader(f))

        assert len(rows) == 2  # header + 1 data row (not 3 split rows)
        assert rows[1][0] == "A, B, and C"
        assert rows[1][1] == "Step 1, Step 2, Step 3"
