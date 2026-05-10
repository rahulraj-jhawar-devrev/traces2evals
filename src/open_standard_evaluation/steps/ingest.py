from typing import Optional

from open_standard_evaluation.adapters.base import BaseAdapter, parse_relative_date
from open_standard_evaluation.checkpoint import CheckpointManager
from open_standard_evaluation.config import SourceConfig
from open_standard_evaluation.models.session import NormalizedSession


def run_ingest(
    adapter: BaseAdapter,
    config: SourceConfig,
    checkpoint: CheckpointManager,
) -> list[NormalizedSession]:
    """Fetch sessions from the adapter, skipping already-ingested ones."""
    already_done = checkpoint.get_completed_sessions("ingested")

    from_date = parse_relative_date(config.langfuse.from_date)
    to_date = parse_relative_date(config.langfuse.to_date)

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
