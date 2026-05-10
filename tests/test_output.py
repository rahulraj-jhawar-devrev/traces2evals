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
    def test_writes_valid_jsonl(self, tmp_path):
        path = tmp_path / "out" / "evals.jsonl"
        write_jsonl([_make_eval()], path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

        row = json.loads(lines[0])
        assert row["input"] == "How do I reset my password?"
        assert row["ideal"] == "Go to Settings > Security > Reset Password."
        assert row["metadata"]["cluster_id"] == 1
        assert row["metadata"]["cluster_title"] == "Account Management"
        assert row["metadata"]["synthesis_type"] == "verbatim"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "evals.jsonl"
        write_jsonl([_make_eval()], path)
        assert path.exists()

    def test_multiple_evals(self, tmp_path):
        evals = [_make_eval(user_message=f"Question {i}") for i in range(5)]
        path = tmp_path / "evals.jsonl"
        write_jsonl(evals, path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 5
        for i, line in enumerate(lines):
            assert json.loads(line)["input"] == f"Question {i}"

    def test_empty_list_writes_empty_file(self, tmp_path):
        path = tmp_path / "evals.jsonl"
        write_jsonl([], path)
        assert path.read_text() == ""

    def test_unicode_content(self, tmp_path):
        eval_item = _make_eval(
            user_message="日本語の質問です",
            reference_response="日本語の回答です",
        )
        path = tmp_path / "evals.jsonl"
        write_jsonl([eval_item], path)

        row = json.loads(path.read_text().strip())
        assert row["input"] == "日本語の質問です"


class TestBraintrustOutput:
    def test_writes_braintrust_format(self, tmp_path):
        path = tmp_path / "evals.jsonl"
        write_braintrust([_make_eval()], path)

        row = json.loads(path.read_text().strip())
        assert row["input"] == "How do I reset my password?"
        assert row["expected"] == "Go to Settings > Security > Reset Password."
        assert "metadata" in row

    def test_metadata_includes_cluster_info(self, tmp_path):
        path = tmp_path / "evals.jsonl"
        write_braintrust([_make_eval()], path)

        row = json.loads(path.read_text().strip())
        assert row["metadata"]["cluster_id"] == 1
        assert row["metadata"]["synthesis_type"] == "verbatim"


class TestCsvOutput:
    def test_writes_csv_with_headers(self, tmp_path):
        path = tmp_path / "evals.csv"
        write_csv([_make_eval()], path)

        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert rows[0] == ["user_message", "reference_output", "remarks"]
        assert rows[1][0] == "How do I reset my password?"
        assert rows[1][1] == "Go to Settings > Security > Reset Password."
        assert rows[1][2] == "Tests account recovery workflow guidance"

    def test_multiple_evals(self, tmp_path):
        evals = [_make_eval(user_message=f"Q{i}") for i in range(3)]
        path = tmp_path / "evals.csv"
        write_csv(evals, path)

        with open(path) as f:
            rows = list(csv.reader(f))

        assert len(rows) == 4  # header + 3 data rows

    def test_csv_handles_commas_in_content(self, tmp_path):
        eval_item = _make_eval(
            user_message="A, B, and C",
            reference_response="Step 1, Step 2, Step 3",
        )
        path = tmp_path / "evals.csv"
        write_csv([eval_item], path)

        with open(path) as f:
            rows = list(csv.reader(f))

        assert rows[1][0] == "A, B, and C"
        assert rows[1][1] == "Step 1, Step 2, Step 3"
