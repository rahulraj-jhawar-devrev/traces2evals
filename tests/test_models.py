from traces2evals.models.session import Message, NormalizedSession, Speaker


class TestNormalizedSession:
    def test_num_turns_counts_user_and_assistant(self, sample_session):
        assert sample_session.num_turns == 4

    def test_num_turns_ignores_system_and_tool(self):
        session = NormalizedSession(
            session_id="s1",
            messages=[
                Message(role=Speaker.SYSTEM, content="You are a helpful assistant."),
                Message(role=Speaker.USER, content="Hello"),
                Message(role=Speaker.TOOL, content='{"result": "data"}'),
                Message(role=Speaker.ASSISTANT, content="Hi there!"),
            ],
        )
        assert session.num_turns == 2

    def test_conversation_text_formats_correctly(self, sample_session):
        text = sample_session.conversation_text
        assert text.startswith("User: How do I reset my password?")
        assert "Assistant: Go to Settings" in text
        assert "User: Thanks" in text

    def test_user_messages_text(self, sample_session):
        text = sample_session.user_messages_text
        assert "How do I reset my password?" in text
        assert "Thanks, that worked!" in text
        assert "Go to Settings" not in text

    def test_empty_session(self):
        session = NormalizedSession(session_id="empty", messages=[])
        assert session.num_turns == 0
        assert session.conversation_text == ""
        assert session.user_messages_text == ""
