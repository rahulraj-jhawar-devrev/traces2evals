from typing import Optional

from open_standard_evaluation.checkpoint import CheckpointManager
from open_standard_evaluation.config import QualityConfig
from open_standard_evaluation.definitions import EnrichmentDefinitions
from open_standard_evaluation.llm.client import LLMClient
from open_standard_evaluation.models.session import NormalizedSession
from open_standard_evaluation.prompts import QUALITY_ASSESSMENT_PROMPT


def run_quality_gate(
    sessions: list[NormalizedSession],
    config: QualityConfig,
    llm: LLMClient,
    checkpoint: CheckpointManager,
    definitions: Optional[EnrichmentDefinitions] = None,
) -> list[NormalizedSession]:
    """Filter sessions by quality. Uses platform scores first, LLM fallback second."""
    already_done = checkpoint.get_completed_sessions("quality_gated")
    to_process = [s for s in sessions if s.session_id not in already_done]
    passed = [s for s in sessions if s.session_id in already_done]

    for session in to_process:
        # Check noise patterns from definitions
        if definitions and _is_noise_from_definitions(session, definitions):
            continue

        quality = _assess_quality(session, config, llm, definitions)
        if quality >= config.min_quality_threshold:
            passed.append(session)

    checkpoint.mark_completed("quality_gated", [s.session_id for s in passed])
    return passed


def _is_noise_from_definitions(
    session: NormalizedSession, definitions: EnrichmentDefinitions
) -> bool:
    """Check if all user messages match noise patterns from definitions."""
    if not definitions.noise_patterns:
        return False

    noise_set = {p.lower().strip() for p in definitions.noise_patterns}
    user_messages = [
        m.content.lower().strip()
        for m in session.messages
        if m.role.value == "user"
    ]

    if not user_messages:
        return True

    return all(msg in noise_set or len(msg) <= 3 for msg in user_messages)


def _assess_quality(
    session: NormalizedSession,
    config: QualityConfig,
    llm: LLMClient,
    definitions: Optional[EnrichmentDefinitions] = None,
) -> float:
    """Return quality score 0-1 for a session."""
    # Tier 1: platform scores with definitions-aware interpretation
    if config.use_platform_scores and session.scores:
        return _score_from_platform(session, config, definitions)

    # Tier 2: LLM assessment
    if config.fallback_to_llm:
        return _score_from_llm(session, llm, definitions)

    return 1.0  # No gate if both tiers disabled


def _score_from_platform(
    session: NormalizedSession,
    config: QualityConfig,
    definitions: Optional[EnrichmentDefinitions] = None,
) -> float:
    """Derive 0-1 quality from platform scores, using definitions for interpretation."""
    for score in session.scores:
        if config.score_name and score.name != config.score_name:
            continue

        # If we have definitions, use metric-specific interpretation
        if definitions:
            metric_def = definitions.get_metric(score.name)
            if metric_def:
                return _interpret_score(score.value, metric_def.scale)

        # Fallback: normalize based on value range
        if score.value <= 1.0:
            return score.value
        return score.value / 5.0

    return 0.5  # No matching score found


def _interpret_score(value: float, scale: str) -> float:
    """Normalize a score to 0-1 based on its declared scale."""
    if scale == "binary" or scale == "0-1":
        return min(max(value, 0.0), 1.0)
    elif scale == "1-5":
        return (value - 1) / 4.0
    elif scale == "1-10":
        return (value - 1) / 9.0
    # Fallback
    if value <= 1.0:
        return value
    return value / 5.0


def _score_from_llm(
    session: NormalizedSession,
    llm: LLMClient,
    definitions: Optional[EnrichmentDefinitions] = None,
) -> float:
    """Use LLM to assess session quality, enriched with definitions context."""
    # Build prompt with optional definitions context
    definitions_context = ""
    if definitions:
        formatted = definitions.format_for_prompt()
        if formatted:
            definitions_context = f"\n\nContext about this agent's metrics:\n{formatted}\n"

    prompt = QUALITY_ASSESSMENT_PROMPT.format(
        conversation_text=session.conversation_text[:8000]
    )

    if definitions_context:
        prompt = prompt + definitions_context

    try:
        result = llm.chat_json(prompt)
        if result.get("is_noise"):
            return 0.0
        quality = result.get("response_quality", 3)
        return quality / 5.0
    except Exception:
        return 0.5
