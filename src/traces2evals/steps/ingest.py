from datetime import datetime
from typing import Optional

from traces2evals.adapters.base import BaseAdapter
from traces2evals.checkpoint import CheckpointManager
from traces2evals.config import SourceConfig
from traces2evals.models.session import NormalizedSession


def run_ingest(
    adapter: BaseAdapter,
    config: SourceConfig,
    checkpoint: CheckpointManager,
) -> list[NormalizedSession]:
    """Fetch sessions from the adapter, skipping already-ingested ones."""
    already_done = checkpoint.get_completed_sessions("ingested")

    from_date = _parse_date(config.langfuse.from_date)
    to_date = _parse_date(config.langfuse.to_date)

    sessions = []
    for session in adapter.fetch_sessions(
        from_date=from_date,
        to_date=to_date,
        max_sessions=config.langfuse.max_sessions,
        min_turns=config.langfuse.min_turns,
    ):
        if session.session_id in already_done:
            continue
        sessions.append(session)

    if sessions:
        checkpoint.mark_completed("ingested", [s.session_id for s in sessions])

    return sessions


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    if value.endswith("d"):
        from datetime import timedelta
        days = int(value[:-1])
        return datetime.utcnow() - timedelta(days=days)
    return datetime.fromisoformat(value)
