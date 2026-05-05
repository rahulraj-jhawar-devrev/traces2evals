from traces2evals.adapters.langfuse import LangfuseAdapter
from traces2evals.models.session import Speaker


class TestLangfuseAdapterContentExtraction:
    """Test the _extract_content method with various input formats."""

    def _make_adapter(self):
        """Create adapter without connecting to Langfuse."""
        adapter = LangfuseAdapter.__new__(LangfuseAdapter)
        return adapter

    def test_extracts_plain_string(self):
        adapter = self._make_adapter()
        assert adapter._extract_content("Hello world") == "Hello world"

    def test_strips_whitespace(self):
        adapter = self._make_adapter()
        assert adapter._extract_content("  hello  ") == "hello"

    def test_returns_none_for_empty_string(self):
        adapter = self._make_adapter()
        assert adapter._extract_content("") is None
        assert adapter._extract_content("   ") is None

    def test_returns_none_for_none(self):
        adapter = self._make_adapter()
        assert adapter._extract_content(None) is None

    def test_extracts_from_dict_text_key(self):
        adapter = self._make_adapter()
        assert adapter._extract_content({"text": "Hello"}) == "Hello"

    def test_extracts_from_dict_content_key(self):
        adapter = self._make_adapter()
        assert adapter._extract_content({"content": "World"}) == "World"

    def test_extracts_from_dict_message_key(self):
        adapter = self._make_adapter()
        assert adapter._extract_content({"message": "Hi"}) == "Hi"

    def test_extracts_from_messages_list_in_dict(self):
        adapter = self._make_adapter()
        value = {"messages": [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Last"},
        ]}
        assert adapter._extract_content(value) == "Last"

    def test_extracts_from_list_of_dicts(self):
        adapter = self._make_adapter()
        value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Response"},
        ]
        assert adapter._extract_content(value) == "Response"

    def test_extracts_from_list_of_strings(self):
        adapter = self._make_adapter()
        assert adapter._extract_content(["first", "second"]) == "second"

    def test_returns_none_for_empty_list(self):
        adapter = self._make_adapter()
        assert adapter._extract_content([]) is None
