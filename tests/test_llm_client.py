from unittest.mock import patch

import pytest

from open_standard_evaluation.llm.client import LLMClient


class TestJsonParsing:
    """LLMs constantly wrap JSON in markdown fences. This is a real parsing bug source."""

    def setup_method(self):
        self.client = LLMClient()

    def test_strips_markdown_fences_with_lang(self):
        text = '```json\n{"key": "value"}\n```'
        result = self.client._parse_json(text)
        assert result == {"key": "value"}

    def test_strips_fences_without_lang(self):
        text = '```\n{"key": "value"}\n```'
        result = self.client._parse_json(text)
        assert result == {"key": "value"}

    def test_strips_leading_whitespace(self):
        text = '  \n  {"key": "value"}  \n  '
        result = self.client._parse_json(text)
        assert result == {"key": "value"}


class TestChatJsonBatchErrorHandling:
    """Before the fix, a single LLM failure killed the entire batch (100+ facets lost).
    Now it should insert None at the failed index and continue."""

    def setup_method(self):
        self.client = LLMClient(max_parallel_calls=2)

    @patch.object(LLMClient, "chat_json")
    def test_one_failure_doesnt_kill_batch(self, mock_chat):
        mock_chat.side_effect = [{"ok": 1}, RuntimeError("API rate limit"), {"ok": 3}]
        results = self.client.chat_json_batch(["a", "b", "c"])
        assert results[0] == {"ok": 1}
        assert results[1] is None
        assert results[2] == {"ok": 3}
        assert len(results) == 3, "output length must always match input length"

    @patch.object(LLMClient, "chat_json")
    def test_all_failures_returns_nones(self, mock_chat):
        """Complete API outage should return Nones, not crash."""
        mock_chat.side_effect = RuntimeError("503 Service Unavailable")
        results = self.client.chat_json_batch(["a", "b"])
        assert results == [None, None]
