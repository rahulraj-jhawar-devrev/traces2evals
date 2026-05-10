"""End-to-end test using a synthetic local adapter.

Runs the full pipeline (ingest → quality gate → facets → embedding →
clustering → labeling → eval generation → coverage gate → output) on
a small in-memory dataset. No external services required.
"""

from datetime import datetime, timezone
from typing import Iterator, Optional

import numpy as np
import pytest

from open_standard_evaluation.adapters.base import BaseAdapter, parse_relative_date
from open_standard_evaluation.checkpoint import CheckpointManager
from open_standard_evaluation.config import (
    ClusteringConfig,
    EvalGenerationConfig,
    FacetsConfig,
    LLMConfig,
    QualityConfig,
    TracesConfig,
)
from open_standard_evaluation.definitions import EnrichmentDefinitions
from open_standard_evaluation.llm.client import LLMClient
from open_standard_evaluation.models.facet import Facet
from open_standard_evaluation.models.session import Message, NormalizedSession, SessionScore, Speaker


# ---------------------------------------------------------------------------
# Synthetic adapter — returns hardcoded sessions
# ---------------------------------------------------------------------------

class SyntheticAdapter(BaseAdapter):
    """In-memory adapter for testing. No network calls."""

    def __init__(self, sessions: list[NormalizedSession]):
        self._sessions = sessions

    def health_check(self) -> bool:
        return True

    def fetch_sessions(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_sessions: int = 5000,
        min_turns: int = 2,
    ) -> Iterator[NormalizedSession]:
        for s in self._sessions:
            if s.num_turns >= min_turns:
                yield s

    def get_session_scores(self, session_id: str) -> list[SessionScore]:
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_session(sid: str, messages: list[Message], score: float = 0.9) -> NormalizedSession:
    return NormalizedSession(
        session_id=sid,
        user_id=f"user_{sid}",
        messages=messages,
        scores=[SessionScore(name="quality", value=score, source="test")],
        created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def synthetic_sessions():
    """15 realistic sessions across 4 behavioral clusters."""
    sessions = [
        # Cluster: Password / Account Recovery (4 sessions)
        _make_session("sess_01", [
            Message(role=Speaker.USER, content="I forgot my password, how do I reset it?"),
            Message(role=Speaker.ASSISTANT, content="Go to Settings > Security > Reset Password. Enter your email and follow the link."),
            Message(role=Speaker.USER, content="Didn't get the email"),
            Message(role=Speaker.ASSISTANT, content="Check your spam folder. If still missing, try requesting again after 60 seconds."),
        ]),
        _make_session("sess_02", [
            Message(role=Speaker.USER, content="Can you help me change my password?"),
            Message(role=Speaker.ASSISTANT, content="Navigate to Settings > Profile > Change Password. Enter your current and new password."),
        ]),
        _make_session("sess_03", [
            Message(role=Speaker.USER, content="I'm locked out of my account"),
            Message(role=Speaker.ASSISTANT, content="Use the 'Forgot Password' link on the login page. If you have 2FA enabled, you'll also need your recovery code."),
        ]),
        _make_session("sess_04", [
            Message(role=Speaker.USER, content="How to reset password for another team member?"),
            Message(role=Speaker.ASSISTANT, content="As an admin, go to Team Management > Members > select the user > Reset Password."),
        ]),
        # Cluster: Billing & Payments (4 sessions)
        _make_session("sess_05", [
            Message(role=Speaker.USER, content="How much does the Pro plan cost?"),
            Message(role=Speaker.ASSISTANT, content="The Pro plan is $29/month billed annually or $39/month billed monthly. It includes unlimited projects and priority support."),
        ]),
        _make_session("sess_06", [
            Message(role=Speaker.USER, content="I need to update my credit card"),
            Message(role=Speaker.ASSISTANT, content="Go to Settings > Billing > Payment Methods > Update. We accept Visa, Mastercard, and Amex."),
        ]),
        _make_session("sess_07", [
            Message(role=Speaker.USER, content="Can I get a refund for last month?"),
            Message(role=Speaker.ASSISTANT, content="We offer a 14-day refund policy. Since last month is beyond that window, I can offer a credit toward your next billing cycle. Want me to set that up?"),
        ]),
        _make_session("sess_08", [
            Message(role=Speaker.USER, content="Where can I find my invoices?"),
            Message(role=Speaker.ASSISTANT, content="Navigate to Settings > Billing > Invoice History. You can download PDF invoices from there."),
        ]),
        # Cluster: Data Export & Integration (4 sessions)
        _make_session("sess_09", [
            Message(role=Speaker.USER, content="How do I export all my data?"),
            Message(role=Speaker.ASSISTANT, content="Go to Settings > Data > Export. You can choose CSV or JSON format. Exports are processed within 24 hours and you'll get an email notification."),
        ]),
        _make_session("sess_10", [
            Message(role=Speaker.USER, content="Can I connect this to Slack?"),
            Message(role=Speaker.ASSISTANT, content="Yes! Go to Integrations > Slack > Connect. You'll need workspace admin permissions. Once connected, you can choose which channels receive notifications."),
        ]),
        _make_session("sess_11", [
            Message(role=Speaker.USER, content="How to set up the API?"),
            Message(role=Speaker.ASSISTANT, content="Generate an API key from Settings > Developers > API Keys. Use the key in your Authorization header. Rate limits are 1000 req/min on Pro."),
        ]),
        _make_session("sess_12", [
            Message(role=Speaker.USER, content="Is there a webhook for new signups?"),
            Message(role=Speaker.ASSISTANT, content="Yes, go to Settings > Developers > Webhooks. Create a new webhook, select the 'user.created' event, and enter your endpoint URL."),
        ]),
        # Cluster: Feature Requests (3 sessions)
        _make_session("sess_13", [
            Message(role=Speaker.USER, content="I need dark mode"),
            Message(role=Speaker.ASSISTANT, content="Dark mode is on our roadmap for Q3. I've added your vote to the feature request. You'll be notified when it ships."),
        ]),
        _make_session("sess_14", [
            Message(role=Speaker.USER, content="Can you add SSO support?"),
            Message(role=Speaker.ASSISTANT, content="SSO via SAML and OIDC is available on Enterprise plan. If you're on Pro, I can help you upgrade. Otherwise, I've noted your interest for our product team."),
        ]),
        _make_session("sess_15", [
            Message(role=Speaker.USER, content="When will mobile app be available?"),
            Message(role=Speaker.ASSISTANT, content="The mobile app is in beta testing. Expected public release is next month. Would you like to join the beta?"),
        ]),
    ]
    return sessions


@pytest.fixture
def synthetic_adapter(synthetic_sessions):
    return SyntheticAdapter(synthetic_sessions)


@pytest.fixture
def tmp_checkpoint(tmp_path):
    return CheckpointManager(cache_dir=str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseRelativeDate:
    def test_none(self):
        assert parse_relative_date(None) is None

    def test_days_offset(self):
        result = parse_relative_date("7d")
        assert result is not None
        assert result.tzinfo is not None  # must be timezone-aware

    def test_iso_string(self):
        result = parse_relative_date("2025-06-01T00:00:00+00:00")
        assert result is not None
        assert result.year == 2025


class TestSyntheticAdapter:
    def test_health_check(self, synthetic_adapter):
        assert synthetic_adapter.health_check() is True

    def test_fetch_all_sessions(self, synthetic_adapter, synthetic_sessions):
        sessions = list(synthetic_adapter.fetch_sessions())
        assert len(sessions) == len(synthetic_sessions)

    def test_min_turns_filter(self, synthetic_sessions):
        short_session = NormalizedSession(
            session_id="short",
            messages=[Message(role=Speaker.USER, content="hi")],
            scores=[],
        )
        adapter = SyntheticAdapter(synthetic_sessions + [short_session])
        sessions = list(adapter.fetch_sessions(min_turns=2))
        assert len(sessions) == len(synthetic_sessions)

    def test_get_session_scores(self, synthetic_adapter):
        assert synthetic_adapter.get_session_scores("sess_01") == []


class TestIngestStep:
    def test_ingest_fresh(self, synthetic_adapter, tmp_checkpoint, synthetic_sessions):
        from open_standard_evaluation.steps.ingest import run_ingest
        from open_standard_evaluation.config import SourceConfig

        config = SourceConfig()
        sessions = run_ingest(synthetic_adapter, config, tmp_checkpoint)
        assert len(sessions) == len(synthetic_sessions)

    def test_ingest_respects_checkpoint(self, synthetic_adapter, tmp_checkpoint, synthetic_sessions):
        from open_standard_evaluation.steps.ingest import run_ingest
        from open_standard_evaluation.config import SourceConfig

        config = SourceConfig()
        # First run
        first = run_ingest(synthetic_adapter, config, tmp_checkpoint)
        assert len(first) == len(synthetic_sessions)

        # Second run — should skip all (checkpoint)
        second = run_ingest(synthetic_adapter, config, tmp_checkpoint)
        assert len(second) == 0


class TestQualityGateStep:
    def test_passes_high_quality_sessions(self, synthetic_sessions, tmp_checkpoint):
        from open_standard_evaluation.steps.quality_gate import run_quality_gate

        config = QualityConfig(min_quality_threshold=0.8)
        llm = LLMClient()
        definitions = EnrichmentDefinitions()

        # Use platform scores (no LLM call needed)
        passed = run_quality_gate(synthetic_sessions, config, llm, tmp_checkpoint, definitions)
        assert len(passed) == len(synthetic_sessions)

    def test_filters_low_quality(self, tmp_checkpoint):
        from open_standard_evaluation.steps.quality_gate import run_quality_gate

        low_session = NormalizedSession(
            session_id="bad",
            messages=[
                Message(role=Speaker.USER, content="bad session"),
                Message(role=Speaker.ASSISTANT, content="bad response"),
            ],
            scores=[SessionScore(name="quality", value=0.2, source="test")],
        )
        config = QualityConfig(min_quality_threshold=0.5, fallback_to_llm=False)
        llm = LLMClient()

        passed = run_quality_gate([low_session], config, llm, tmp_checkpoint)
        assert len(passed) == 0


class TestFacetExtraction:
    def test_facet_extraction_produces_facets(self, synthetic_sessions, tmp_checkpoint):
        from open_standard_evaluation.steps.facets import run_facet_extraction

        config = FacetsConfig()
        llm = LLMClient()

        # Facet extraction requires LLM — test the structure, not the content
        # by mocking the LLM
        with pytest.MonkeyPatch.context() as mp:
            def fake_json_batch(prompts, desc=""):
                return [{"facet_summary": f"Summary for session about passwords", "key_topics": ["auth"]} for _ in prompts]
            llm.chat_json_batch = fake_json_batch

            facets = run_facet_extraction(synthetic_sessions, config, llm, tmp_checkpoint)
            assert len(facets) == len(synthetic_sessions)
            assert all(isinstance(f, Facet) for f in facets)
            assert all(f.summary for f in facets)


class TestEmbeddingStep:
    def test_embedding_shape(self, tmp_checkpoint):
        from open_standard_evaluation.steps.embed import run_embedding

        facets = [
            Facet(session_id=f"s{i}", facet_idx=0, summary=f"Test facet {i}", num_turns=2)
            for i in range(5)
        ]
        llm = LLMClient()

        with pytest.MonkeyPatch.context() as mp:
            def fake_embed(texts, batch_size=100):
                return np.random.randn(len(texts), 768).astype(np.float32)
            llm.embed = fake_embed

            embeddings = run_embedding(facets, llm, tmp_checkpoint)
            assert embeddings.shape == (5, 768)

    def test_uses_cache_on_second_run(self, tmp_checkpoint):
        from open_standard_evaluation.steps.embed import run_embedding

        facets = [
            Facet(session_id="s1", facet_idx=0, summary="Test facet", num_turns=2),
        ]
        llm = LLMClient(embedding_model="test-model")

        call_count = 0

        def fake_embed(texts, batch_size=100):
            nonlocal call_count
            call_count += 1
            return np.random.randn(len(texts), 768).astype(np.float32)

        llm.embed = fake_embed

        # First run
        run_embedding(facets, llm, tmp_checkpoint)
        # Second run — should load from cache
        run_embedding(facets, llm, tmp_checkpoint)
        assert call_count == 1


class TestClusteringStep:
    def test_clustering_produces_labels(self, synthetic_sessions, tmp_checkpoint):
        from open_standard_evaluation.steps.cluster import run_clustering

        facets = [
            Facet(session_id=s.session_id, facet_idx=0, summary=s.conversation_text[:200], num_turns=s.num_turns)
            for s in synthetic_sessions
        ]
        np.random.seed(42)
        embeddings = np.random.randn(len(facets), 50).astype(np.float32)

        config = ClusteringConfig(
            min_cluster_size=2,
            min_samples=1,
            umap_n_components=10,
            merge_similarity_threshold=0.99,
        )

        labels, n_clusters, hierarchy = run_clustering(facets, embeddings, config)
        assert len(labels) == len(facets)
        assert n_clusters >= 1


class TestCoverageGate:
    def test_accepts_first_eval(self, sample_eval_items):
        from open_standard_evaluation.steps.coverage_gate import run_coverage_gate
        from open_standard_evaluation.models.pattern import PatternIndex

        llm = LLMClient()
        index = PatternIndex()

        with pytest.MonkeyPatch.context() as mp:
            llm.chat_json = lambda prompt: {
                "passes_gate": True,
                "reason": "New pattern",
                "matched_pattern": None,
                "suggested_pattern": "Account Recovery",
                "is_duplicate_query": False,
                "capability_category": "account recovery",
            }

            accepted, updated = run_coverage_gate(sample_eval_items, index, llm)
            assert len(accepted) == 1
            assert len(updated.patterns) == 1


class TestOutputStep:
    def test_jsonl_round_trip(self, tmp_path, sample_eval_items):
        from open_standard_evaluation.output.jsonl import write_jsonl
        import json

        path = tmp_path / "evals.jsonl"
        write_jsonl(sample_eval_items, path)

        rows = [json.loads(line) for line in path.read_text().strip().split("\n")]
        assert len(rows) == len(sample_eval_items)
        assert "input" in rows[0]
        assert "ideal" in rows[0]


# ---------------------------------------------------------------------------
# Full pipeline E2E
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Run the full pipeline on synthetic data (no LLM calls)."""

    def test_pipeline_produces_output(self, tmp_path, synthetic_sessions):
        from open_standard_evaluation.output.jsonl import write_jsonl
        from open_standard_evaluation.steps.coverage_gate import load_pattern_index, run_coverage_gate, save_pattern_index
        from open_standard_evaluation.steps.ingest import run_ingest
        from open_standard_evaluation.steps.quality_gate import run_quality_gate
        from open_standard_evaluation.models.pattern import PatternIndex

        adapter = SyntheticAdapter(synthetic_sessions)
        checkpoint = CheckpointManager(cache_dir=str(tmp_path / "cache"))
        config = TracesConfig()
        config.quality.use_platform_scores = True
        config.quality.fallback_to_llm = False
        config.quality.min_quality_threshold = 0.5
        config.eval_generation.coverage_gate = False

        # Step 1: Ingest
        sessions = run_ingest(adapter, config.source, checkpoint)
        assert len(sessions) == 15

        # Step 2: Quality gate
        sessions = run_quality_gate(sessions, config.quality, LLMClient(), checkpoint)
        assert len(sessions) > 0

        # Step 3-7: Skip (need LLM). Verify output with pre-built evals.
        from open_standard_evaluation.models.eval_item import EvalItem

        evals = [
            EvalItem(
                user_message=sessions[i].messages[0].content,
                reference_response=sessions[i].messages[1].content,
                judge_remarks="E2E test eval",
                cluster_id=i % 4,
                cluster_title=["Account Recovery", "Billing", "Data Export", "Feature Requests"][i % 4],
                source_session_id=sessions[i].session_id,
                synthesis_type="verbatim",
            )
            for i in range(len(sessions))
        ]

        # Step 8: Coverage gate
        pattern_index = PatternIndex()
        pattern_path = tmp_path / "pattern_index.json"

        with pytest.MonkeyPatch.context() as mp:
            LLMClient.chat_json = lambda self, prompt: {
                "passes_gate": True,
                "reason": "New pattern",
                "matched_pattern": None,
                "suggested_pattern": f"Pattern {i}",
                "is_duplicate_query": False,
                "capability_category": "test",
            }

            llm = LLMClient()
            accepted, updated = run_coverage_gate(evals, pattern_index, llm)
            save_pattern_index(updated, pattern_path)

        assert len(accepted) > 0

        # Step 9: Output
        output_dir = tmp_path / "output"
        write_jsonl(accepted, output_dir / "evals.jsonl")
        assert (output_dir / "evals.jsonl").exists()
