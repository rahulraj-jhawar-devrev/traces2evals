"""Integration tests for pipeline steps using a synthetic adapter.

No external services. Tests real step logic: checkpointing, filtering,
caching, and error recovery — not mock choreography."""

from datetime import datetime, timezone
from typing import Iterator, Optional

import numpy as np
import pytest

from open_standard_evaluation.adapters.base import BaseAdapter, parse_relative_date
from open_standard_evaluation.checkpoint import CheckpointManager
from open_standard_evaluation.config import QualityConfig, SourceConfig
from open_standard_evaluation.definitions import EnrichmentDefinitions
from open_standard_evaluation.llm.client import LLMClient
from open_standard_evaluation.models.facet import Facet
from open_standard_evaluation.models.session import Message, NormalizedSession, SessionScore, Speaker


class SyntheticAdapter(BaseAdapter):
    def __init__(self, sessions: list[NormalizedSession]):
        self._sessions = sessions

    def health_check(self) -> bool:
        return True

    def fetch_sessions(self, from_date=None, to_date=None, max_sessions=5000, min_turns=2):
        for s in self._sessions:
            if s.num_turns >= min_turns:
                yield s

    def get_session_scores(self, session_id: str):
        return []


def _make_session(sid: str, messages: list[Message], score: float = 0.9) -> NormalizedSession:
    return NormalizedSession(
        session_id=sid,
        user_id=f"user_{sid}",
        messages=messages,
        scores=[SessionScore(name="quality", value=score, source="test")],
        created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def sessions():
    return [
        _make_session("sess_01", [
            Message(role=Speaker.USER, content="How do I reset my password?"),
            Message(role=Speaker.ASSISTANT, content="Go to Settings > Security > Reset Password."),
        ]),
        _make_session("sess_02", [
            Message(role=Speaker.USER, content="What's the Pro plan price?"),
            Message(role=Speaker.ASSISTANT, content="$29/month billed annually."),
        ]),
        _make_session("bad_session", [
            Message(role=Speaker.USER, content="bad"),
            Message(role=Speaker.ASSISTANT, content="bad"),
        ], score=0.2),
    ]


# --- parse_relative_date ---

class TestParseRelativeDate:
    def test_none_returns_none(self):
        assert parse_relative_date(None) is None

    def test_relative_offset_is_timezone_aware(self):
        result = parse_relative_date("7d")
        assert result.tzinfo is not None

    def test_iso_string(self):
        result = parse_relative_date("2025-06-01T00:00:00+00:00")
        assert result.year == 2025


# --- Ingest ---

class TestIngest:
    def test_fresh_ingest(self, sessions, tmp_path):
        from open_standard_evaluation.steps.ingest import run_ingest

        adapter = SyntheticAdapter(sessions)
        checkpoint = CheckpointManager(cache_dir=str(tmp_path / "cache"))
        ingested = run_ingest(adapter, SourceConfig(), checkpoint)
        assert len(ingested) == 3

    def test_checkpoint_skips_already_ingested(self, sessions, tmp_path):
        from open_standard_evaluation.steps.ingest import run_ingest

        adapter = SyntheticAdapter(sessions)
        checkpoint = CheckpointManager(cache_dir=str(tmp_path / "cache"))

        first = run_ingest(adapter, SourceConfig(), checkpoint)
        second = run_ingest(adapter, SourceConfig(), checkpoint)
        assert len(first) == 3
        assert len(second) == 0, "already-ingested sessions should be skipped"

    def test_min_turns_filter(self):
        from open_standard_evaluation.steps.ingest import run_ingest

        short = NormalizedSession(
            session_id="short", messages=[Message(role=Speaker.USER, content="hi")], scores=[]
        )
        ok = _make_session("ok", [
            Message(role=Speaker.USER, content="q"),
            Message(role=Speaker.ASSISTANT, content="a"),
        ])
        adapter = SyntheticAdapter([short, ok])
        fetched = list(adapter.fetch_sessions(min_turns=2))
        assert len(fetched) == 1
        assert fetched[0].session_id == "ok"


# --- Quality Gate ---

class TestQualityGate:
    def test_filters_by_platform_score(self, sessions, tmp_path):
        from open_standard_evaluation.steps.quality_gate import run_quality_gate

        checkpoint = CheckpointManager(cache_dir=str(tmp_path / "cache"))
        config = QualityConfig(min_quality_threshold=0.5, fallback_to_llm=False)
        passed = run_quality_gate(sessions, config, LLMClient(), checkpoint, EnrichmentDefinitions())
        assert len(passed) == 2, "bad_session (score=0.2) should be filtered out"

    def test_all_pass_when_below_threshold(self, sessions, tmp_path):
        from open_standard_evaluation.steps.quality_gate import run_quality_gate

        checkpoint = CheckpointManager(cache_dir=str(tmp_path / "cache"))
        config = QualityConfig(min_quality_threshold=0.1, fallback_to_llm=False)
        passed = run_quality_gate(sessions, config, LLMClient(), checkpoint, EnrichmentDefinitions())
        assert len(passed) == 3


# --- Embedding Cache ---

class TestEmbeddingCache:
    def test_second_run_uses_cache(self, tmp_path):
        from open_standard_evaluation.steps.embed import run_embedding

        facets = [Facet(session_id="s1", facet_idx=0, summary="Test", num_turns=2)]
        checkpoint = CheckpointManager(cache_dir=str(tmp_path / "cache"))
        llm = LLMClient(embedding_model="test-model")

        call_count = 0
        def fake_embed(texts, batch_size=100):
            nonlocal call_count
            call_count += 1
            return np.random.randn(len(texts), 768).astype(np.float32)
        llm.embed = fake_embed

        run_embedding(facets, llm, checkpoint)
        run_embedding(facets, llm, checkpoint)
        assert call_count == 1, "second run should load from cache, not re-embed"

    def test_wrong_model_bypasses_cache(self, tmp_path):
        """If embedding model changes, cached embeddings must not be reused.
        Mismatched model embeddings would silently produce wrong evals."""
        checkpoint = CheckpointManager(cache_dir=str(tmp_path / "cache"))

        # Simulate cache from model-a
        np.save(tmp_path / "cache" / "embeddings.npy", np.random.randn(1, 768).astype(np.float32))
        checkpoint.save_meta({"embedding_model": "model-a"})

        # model-b should NOT get model-a's cache
        result = checkpoint.load_embeddings(1, "model-b")
        assert result is None

        # model-a SHOULD get it
        result = checkpoint.load_embeddings(1, "model-a")
        assert result is not None
        assert result.shape == (1, 768)


# --- Corrupted Cache ---

class TestCorruptedCache:
    def test_corrupted_manifest_starts_fresh(self, tmp_path):
        checkpoint = CheckpointManager(cache_dir=str(tmp_path / "cache"))
        checkpoint.mark_completed("ingested", ["s1"])
        # Corrupt the manifest
        (tmp_path / "cache" / "manifest.json").write_text("NOT VALID JSON{{{")

        checkpoint2 = CheckpointManager(cache_dir=str(tmp_path / "cache"))
        assert checkpoint2.get_completed_sessions("ingested") == set(), "corrupted manifest should not crash, should start fresh"

    def test_corrupted_embeddings_returns_none(self, tmp_path):
        checkpoint = CheckpointManager(cache_dir=str(tmp_path / "cache"))
        # Write garbage as embeddings
        (tmp_path / "cache" / "embeddings.npy").write_bytes(b"not a numpy array")
        checkpoint.save_meta({"embedding_model": "test"})

        result = checkpoint.load_embeddings(5, "test")
        assert result is None, "corrupted .npy should return None, not crash"
