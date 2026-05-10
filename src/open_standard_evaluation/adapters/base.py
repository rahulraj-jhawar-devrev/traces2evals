from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

from open_standard_evaluation.models.session import NormalizedSession, SessionScore


def parse_relative_date(value: Optional[str]) -> Optional[datetime]:
    """Parse a date string or relative offset (e.g. '30d') into a datetime.

    Supports ISO-8601 strings and relative day offsets like '7d', '30d'.
    Returns None if value is None.
    """
    if value is None:
        return None
    if value.endswith("d"):
        days = int(value[:-1])
        return datetime.now(timezone.utc) - timedelta(days=days)
    return datetime.fromisoformat(value)


class BaseAdapter(ABC):
    """
    Abstract interface for observability platform adapters.

    Implement this to add support for a new platform (Opik, Phoenix, etc.).
    Each adapter handles: connection, session fetching, normalization, and score extraction.
    """

    @abstractmethod
    def fetch_sessions(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_sessions: int = 5000,
        min_turns: int = 2,
    ) -> Iterator[NormalizedSession]:
        """
        Yield normalized sessions from the platform.

        Handles pagination internally. Yields one session at a time
        to support streaming without OOM on large instances.
        """
        ...

    @abstractmethod
    def get_session_scores(self, session_id: str) -> list[SessionScore]:
        """
        Fetch platform-native quality scores for a session.
        Returns empty list if no scores are available.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Verify connectivity and credentials. Raise on failure."""
        ...
