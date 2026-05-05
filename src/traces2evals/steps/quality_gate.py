from traces2evals.checkpoint import CheckpointManager
from traces2evals.config import QualityConfig
from traces2evals.llm.client import LLMClient
from traces2evals.models.session import NormalizedSession
from traces2evals.prompts import QUALITY_ASSESSMENT_PROMPT


def run_quality_gate(
    sessions: list[NormalizedSession],
    config: QualityConfig,
    llm: LLMClient,
    checkpoint: CheckpointManager,
) -> list[NormalizedSession]:
    """Filter sessions by quality. Uses platform scores first, LLM fallback second."""
    already_done = checkpoint.get_completed_sessions("quality_gated")
    to_process = [s for s in sessions if s.session_id not in already_done]
    passed = [s for s in sessions if s.session_id in already_done]

    for session in to_process:
        quality = _assess_quality(session, config, llm)
        if quality >= config.min_quality_threshold:
            passed.append(session)

    checkpoint.mark_completed("quality_gated", [s.session_id for s in passed])
    return passed


def _assess_quality(
    session: NormalizedSession, config: QualityConfig, llm: LLMClient
) -> float:
    """Return quality score 0-1 for a session."""
    # Tier 1: platform scores
    if config.use_platform_scores and session.scores:
        return _score_from_platform(session, config)

    # Tier 2: LLM assessment
    if config.fallback_to_llm:
        return _score_from_llm(session, llm)

    return 1.0  # No gate if both tiers disabled


def _score_from_platform(session: NormalizedSession, config: QualityConfig) -> float:
    """Derive 0-1 quality from platform scores."""
    for score in session.scores:
        if config.score_name and score.name != config.score_name:
            continue
        # Normalize: if scale is 0-1 use directly, if 1-5 normalize
        if score.value <= 1.0:
            return score.value
        return score.value / 5.0
    return 0.5  # No matching score found


def _score_from_llm(session: NormalizedSession, llm: LLMClient) -> float:
    """Use LLM to assess session quality."""
    prompt = QUALITY_ASSESSMENT_PROMPT.format(
        conversation_text=session.conversation_text[:8000]
    )
    try:
        result = llm.chat_json(prompt)
        if result.get("is_noise"):
            return 0.0
        quality = result.get("response_quality", 3)
        return quality / 5.0
    except Exception:
        return 0.5
