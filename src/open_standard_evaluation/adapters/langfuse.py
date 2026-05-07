from datetime import datetime
from typing import Iterator, Optional

from open_standard_evaluation.adapters.base import BaseAdapter
from open_standard_evaluation.config import LangfuseSourceConfig
from open_standard_evaluation.models.session import Message, NormalizedSession, SessionScore, Speaker


class LangfuseAdapter(BaseAdapter):
    """
    Langfuse adapter — fetches sessions via the Langfuse Python SDK.

    Data model mapping:
    - Langfuse Session → NormalizedSession
    - Langfuse Traces (ordered by timestamp) → Messages
    - Langfuse Scores → SessionScores
    """

    def __init__(self, config: LangfuseSourceConfig):
        from langfuse import Langfuse

        self.config = config
        self.client = Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            host=config.host,
        )

    def health_check(self) -> bool:
        """Verify credentials by attempting to list one session."""
        try:
            self.client.fetch_sessions(limit=1)
            return True
        except Exception as e:
            raise ConnectionError(f"Langfuse health check failed: {e}") from e

    def fetch_sessions(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_sessions: int = 5000,
        min_turns: int = 2,
    ) -> Iterator[NormalizedSession]:
        """Fetch and normalize sessions from Langfuse."""
        page = 1
        yielded = 0

        while yielded < max_sessions:
            response = self.client.fetch_sessions(
                limit=50,
                page=page,
                from_timestamp=from_date,
                to_timestamp=to_date,
            )
            sessions = response.data
            if not sessions:
                break

            for session in sessions:
                if yielded >= max_sessions:
                    return

                normalized = self._normalize_session(session, min_turns)
                if normalized is not None:
                    yielded += 1
                    yield normalized

            page += 1

    def _normalize_session(
        self, session, min_turns: int
    ) -> Optional[NormalizedSession]:
        """Convert a Langfuse session + its traces into a NormalizedSession."""
        traces = self._fetch_traces_for_session(session.id)
        if not traces:
            return None

        messages = self._extract_messages(traces)
        if len([m for m in messages if m.role in (Speaker.USER, Speaker.ASSISTANT)]) < min_turns:
            return None

        scores = self._extract_scores(traces)
        user_id = self._extract_user_id(traces)

        return NormalizedSession(
            session_id=session.id,
            user_id=user_id,
            messages=messages,
            scores=scores,
            metadata={"trace_count": len(traces)},
            created_at=getattr(session, "created_at", None),
        )

    def _fetch_traces_for_session(self, session_id: str) -> list:
        """Paginate through all traces for a session, ordered by timestamp."""
        all_traces = []
        page = 1
        while True:
            response = self.client.fetch_traces(
                session_id=session_id,
                limit=100,
                page=page,
                order_by="timestamp.asc",
            )
            traces = response.data
            if not traces:
                break
            all_traces.extend(traces)
            page += 1
        return all_traces

    def _extract_messages(self, traces: list) -> list[Message]:
        """
        Extract user/assistant messages from traces.

        Strategy: each trace typically represents one exchange.
        - trace.input → user message
        - trace.output → assistant message
        Handle both string and dict formats.
        """
        messages = []
        seen_contents = set()

        for trace in traces:
            user_content = self._extract_content(getattr(trace, "input", None))
            assistant_content = self._extract_content(getattr(trace, "output", None))
            timestamp = getattr(trace, "timestamp", None)

            if user_content and user_content not in seen_contents:
                seen_contents.add(user_content)
                messages.append(Message(
                    role=Speaker.USER,
                    content=user_content,
                    timestamp=timestamp,
                ))

            if assistant_content and assistant_content not in seen_contents:
                seen_contents.add(assistant_content)
                messages.append(Message(
                    role=Speaker.ASSISTANT,
                    content=assistant_content,
                    timestamp=timestamp,
                ))

        return messages

    def _extract_content(self, value) -> Optional[str]:
        """Extract string content from trace input/output (handles str, dict, list)."""
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip() if value.strip() else None
        if isinstance(value, dict):
            # Common patterns: {"text": "..."}, {"content": "..."}, {"message": "..."}
            for key in ("text", "content", "message", "output"):
                if key in value and isinstance(value[key], str):
                    return value[key].strip() if value[key].strip() else None
            # Chat history format: last message
            if "messages" in value and isinstance(value["messages"], list):
                msgs = value["messages"]
                if msgs:
                    last = msgs[-1]
                    if isinstance(last, dict) and "content" in last:
                        return last["content"].strip() if last["content"] else None
        if isinstance(value, list) and value:
            last = value[-1]
            if isinstance(last, dict) and "content" in last:
                return last["content"].strip() if last["content"] else None
            if isinstance(last, str):
                return last.strip() if last.strip() else None
        return None

    def _extract_scores(self, traces: list) -> list[SessionScore]:
        """Collect scores from all traces in the session."""
        scores = []
        for trace in traces:
            trace_id = getattr(trace, "id", None)
            if not trace_id:
                continue
            try:
                response = self.client.fetch_scores(trace_id=trace_id)
                for score in response.data:
                    scores.append(SessionScore(
                        name=score.name,
                        value=float(score.value),
                        source=getattr(score, "source", "unknown"),
                        comment=getattr(score, "comment", None),
                    ))
            except Exception:
                continue
        return scores

    def _extract_user_id(self, traces: list) -> Optional[str]:
        """Get user_id from the first trace that has one."""
        for trace in traces:
            uid = getattr(trace, "user_id", None)
            if uid:
                return uid
        return None

    def get_session_scores(self, session_id: str) -> list[SessionScore]:
        """Fetch scores for a specific session."""
        traces = self._fetch_traces_for_session(session_id)
        return self._extract_scores(traces)
