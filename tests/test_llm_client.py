from traces2evals.llm.client import LLMClient


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
