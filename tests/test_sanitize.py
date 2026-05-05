from traces2evals.models.eval_item import EvalItem
from traces2evals.steps.sanitize import (
    TEMPORAL_DRIFT_NOTE,
    _contains_temporal_data,
    sanitize_eval_items,
    sanitize_response,
)


class TestSanitizeResponse:
    def test_removes_s3_signed_url(self):
        text = "Here's your report: https://mybucket.s3.us-east-1.amazonaws.com/docs/report.pdf?X-Amz-Signature=abc123&X-Amz-Credential=xyz"
        cleaned, modified = sanitize_response(text)
        assert "[Generated document link]" in cleaned
        assert "X-Amz-Signature" not in cleaned
        assert modified is True

    def test_removes_s3_url(self):
        text = "Download: https://mybucket.s3.us-east-1.amazonaws.com/file.pdf"
        cleaned, modified = sanitize_response(text)
        assert "[Generated document link]" in cleaned
        assert modified is True

    def test_removes_gcs_url(self):
        text = "File at https://storage.googleapis.com/bucket/object.json"
        cleaned, modified = sanitize_response(text)
        assert "[Generated document link]" in cleaned
        assert modified is True

    def test_removes_base64_data(self):
        b64 = "A" * 200
        text = f"Image: data:image/png;base64,{b64}"
        cleaned, modified = sanitize_response(text)
        assert "[Embedded content]" in cleaned
        assert b64 not in cleaned
        assert modified is True

    def test_removes_bearer_token(self):
        text = "Use Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        cleaned, modified = sanitize_response(text)
        assert "[Redacted token]" in cleaned
        assert "eyJ" not in cleaned
        assert modified is True

    def test_leaves_normal_text_unchanged(self):
        text = "The account has 5 open tickets. Please check the dashboard for details."
        cleaned, modified = sanitize_response(text)
        assert cleaned == text
        assert modified is False

    def test_leaves_normal_urls_unchanged(self):
        text = "Visit https://docs.example.com/guide for more info."
        cleaned, modified = sanitize_response(text)
        assert cleaned == text
        assert modified is False


class TestContainsTemporalData:
    def test_detects_iso_date(self):
        assert _contains_temporal_data("Last updated on 2025-03-15")

    def test_detects_slash_date(self):
        assert _contains_temporal_data("Created 3/15/2025")

    def test_detects_temporal_words(self):
        assert _contains_temporal_data("The ticket was created today")
        assert _contains_temporal_data("Currently there are 5 open items")

    def test_detects_dollar_amounts(self):
        assert _contains_temporal_data("Pipeline value: $1,250,000")

    def test_detects_counts(self):
        assert _contains_temporal_data("You have 42 open tickets")

    def test_no_temporal_data(self):
        assert not _contains_temporal_data("Go to Settings and click Export")


class TestSanitizeEvalItems:
    def test_appends_temporal_drift_note(self):
        items = [EvalItem(
            user_message="What's the status?",
            reference_response="There are currently 15 open tickets as of 2025-03-15.",
            judge_remarks="Tests status reporting capability.",
            cluster_id=1,
            cluster_title="Status",
            source_session_id="s1",
            synthesis_type="verbatim",
        )]
        sanitized = sanitize_eval_items(items)
        assert TEMPORAL_DRIFT_NOTE in sanitized[0].judge_remarks

    def test_does_not_duplicate_note(self):
        items = [EvalItem(
            user_message="What's the status?",
            reference_response="There are currently 15 open tickets.",
            judge_remarks=f"Tests status. {TEMPORAL_DRIFT_NOTE}",
            cluster_id=1,
            cluster_title="Status",
            source_session_id="s1",
            synthesis_type="verbatim",
        )]
        sanitized = sanitize_eval_items(items)
        assert sanitized[0].judge_remarks.count(TEMPORAL_DRIFT_NOTE) == 1

    def test_cleans_urls_in_reference(self):
        items = [EvalItem(
            user_message="Generate the report",
            reference_response="Here's your report: https://bucket.s3.us-east-1.amazonaws.com/report.pdf?X-Amz-Signature=abc",
            judge_remarks="Tests report generation.",
            cluster_id=1,
            cluster_title="Reports",
            source_session_id="s1",
            synthesis_type="synthesized",
        )]
        sanitized = sanitize_eval_items(items)
        assert "[Generated document link]" in sanitized[0].reference_response
        assert "X-Amz-Signature" not in sanitized[0].reference_response
