from unittest.mock import patch

import pytest

from open_standard_evaluation.llm.client import LLMClient


class TestJsonParsing:
    """Test the _parse_json helper without making API calls."""

    def setup_method(self):
        self.client = LLMClient()

    def test_parses_plain_json(self):
        result = self.client._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_strips_markdown_fences(self):
        text = '```json\n{"key": "value"}\n```'
        result = self.client._parse_json(text)
        assert result == {"key": "value"}

    def test_strips_fences_without_language(self):
        text = '```\n{"key": "value"}\n```'
        result = self.client._parse_json(text)
        assert result == {"key": "value"}

    def test_handles_nested_json(self):
        text = '{"a": {"b": [1, 2, 3]}, "c": true}'
        result = self.client._parse_json(text)
        assert result == {"a": {"b": [1, 2, 3]}, "c": True}

    def test_handles_whitespace(self):
        text = '  \n  {"key": "value"}  \n  '
        result = self.client._parse_json(text)
        assert result == {"key": "value"}


class TestChatJsonBatch:
    """Test batch error handling in chat_json_batch."""

    def setup_method(self):
        self.client = LLMClient(max_parallel_calls=2)

    @patch.object(LLMClient, "chat_json")
    def test_all_succeed(self, mock_chat):
        mock_chat.side_effect = [{"ok": True}, {"ok": True}]
        results = self.client.chat_json_batch(["prompt1", "prompt2"])
        assert results == [{"ok": True}, {"ok": True}]

    @patch.object(LLMClient, "chat_json")
    def test_one_failure_returns_none_at_index(self, mock_chat):
        """A failed item should produce None at its position, not crash the batch."""
        mock_chat.side_effect = [{"ok": 1}, RuntimeError("API error"), {"ok": 3}]
        results = self.client.chat_json_batch(["a", "b", "c"])
        assert results[0] == {"ok": 1}
        assert results[1] is None
        assert results[2] == {"ok": 3}

    @patch.object(LLMClient, "chat_json")
    def test_all_failures(self, mock_chat):
        """All failures should return all-None list."""
        mock_chat.side_effect = RuntimeError("down")
        results = self.client.chat_json_batch(["a", "b"])
        assert results == [None, None]

    @patch.object(LLMClient, "chat_json")
    def test_empty_input(self, mock_chat):
        results = self.client.chat_json_batch([])
        assert results == []

    @patch.object(LLMClient, "chat_json")
    def test_result_count_matches_input_count(self, mock_chat):
        """Output length must always equal input length."""
        mock_chat.side_effect = [{"ok": i} for i in range(10)]
        results = self.client.chat_json_batch([f"p{i}" for i in range(10)])
        assert len(results) == 10
