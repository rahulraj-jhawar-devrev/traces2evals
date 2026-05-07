from datetime import datetime

import pytest

from open_standard_evaluation.config import TracesConfig
from open_standard_evaluation.models.eval_item import EvalItem
from open_standard_evaluation.models.facet import Facet
from open_standard_evaluation.models.session import Message, NormalizedSession, SessionScore, Speaker


@pytest.fixture
def sample_messages():
    return [
        Message(role=Speaker.USER, content="How do I reset my password?"),
        Message(role=Speaker.ASSISTANT, content="Go to Settings > Security > Reset Password."),
        Message(role=Speaker.USER, content="Thanks, that worked!"),
        Message(role=Speaker.ASSISTANT, content="Happy to help!"),
    ]


@pytest.fixture
def sample_session(sample_messages):
    return NormalizedSession(
        session_id="sess_001",
        user_id="user_42",
        messages=sample_messages,
        scores=[SessionScore(name="quality", value=0.9, source="annotation")],
        created_at=datetime(2025, 3, 15),
    )


@pytest.fixture
def sample_sessions(sample_session):
    return [
        sample_session,
        NormalizedSession(
            session_id="sess_002",
            user_id="user_43",
            messages=[
                Message(role=Speaker.USER, content="Can you explain billing?"),
                Message(role=Speaker.ASSISTANT, content="Your plan is billed monthly..."),
            ],
            scores=[],
        ),
        NormalizedSession(
            session_id="sess_003",
            user_id="user_44",
            messages=[
                Message(role=Speaker.USER, content="How do I export data?"),
                Message(role=Speaker.ASSISTANT, content="Navigate to Data > Export and choose CSV or JSON."),
            ],
            scores=[SessionScore(name="quality", value=0.3, source="annotation")],
        ),
    ]


@pytest.fixture
def sample_facets():
    return [
        Facet(session_id="sess_001", facet_idx=0, summary="User asks how to reset account password", num_turns=4),
        Facet(session_id="sess_002", facet_idx=0, summary="User asks about billing plan details", num_turns=2),
        Facet(session_id="sess_003", facet_idx=0, summary="User asks how to export data from the platform", num_turns=2),
    ]


@pytest.fixture
def sample_eval_items():
    return [
        EvalItem(
            user_message="How do I reset my password?",
            reference_response="Go to Settings > Security > Reset Password.",
            judge_remarks="Tests account recovery workflow guidance",
            cluster_id=1,
            cluster_title="Account Management",
            source_session_id="sess_001",
            synthesis_type="verbatim",
            capability_tested="account recovery",
        ),
    ]


@pytest.fixture
def default_config():
    return TracesConfig()


@pytest.fixture
def tmp_cache(tmp_path):
    return str(tmp_path / "cache")
