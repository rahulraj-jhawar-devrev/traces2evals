import csv
import json

from traces2evals.output.braintrust import write_braintrust
from traces2evals.output.csv import write_csv
from traces2evals.output.jsonl import write_jsonl


class TestJsonlOutput:
    def test_writes_valid_jsonl(self, tmp_path, sample_eval_items):
        path = tmp_path / "out" / "evals.jsonl"
        write_jsonl(sample_eval_items, path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

        row = json.loads(lines[0])
        assert row["input"] == "How do I reset my password?"
        assert row["ideal"] == "Go to Settings > Security > Reset Password."
        assert row["metadata"]["cluster_id"] == 1
        assert row["metadata"]["cluster_title"] == "Account Management"
        assert row["metadata"]["synthesis_type"] == "verbatim"

    def test_creates_parent_dirs(self, tmp_path, sample_eval_items):
        path = tmp_path / "deep" / "nested" / "evals.jsonl"
        write_jsonl(sample_eval_items, path)
        assert path.exists()


class TestBraintrustOutput:
    def test_writes_braintrust_format(self, tmp_path, sample_eval_items):
        path = tmp_path / "evals.jsonl"
        write_braintrust(sample_eval_items, path)

        row = json.loads(path.read_text().strip())
        assert row["input"] == "How do I reset my password?"
        assert row["expected"] == "Go to Settings > Security > Reset Password."
        assert "metadata" in row


class TestCsvOutput:
    def test_writes_csv_with_headers(self, tmp_path, sample_eval_items):
        path = tmp_path / "evals.csv"
        write_csv(sample_eval_items, path)

        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert rows[0] == ["user_message", "reference_output", "remarks"]
        assert rows[1][0] == "How do I reset my password?"
        assert rows[1][1] == "Go to Settings > Security > Reset Password."
