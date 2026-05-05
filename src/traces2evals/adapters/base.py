from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterator, Optional

from traces2evals.models.session import NormalizedSession, SessionScore


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
