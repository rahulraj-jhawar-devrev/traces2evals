"""
Coverage gate — prevents redundant evals by tracking patterns.

Before an eval candidate is accepted, it must pass:
1. Tests something new (not redundant with existing pattern evals)
2. Regression would matter (non-trivial behavior)
3. Distinct capability (different skill/knowledge than other evals)
4. Clean representative example

Maintains a PatternIndex that grows across runs.
"""

import json
from pathlib import Path
from typing import Optional

from open_standard_evaluation.llm.client import LLMClient
from open_standard_evaluation.models.eval_item import EvalItem
from open_standard_evaluation.models.pattern import PatternEntry, PatternIndex

COVERAGE_GATE_PROMPT = """You are evaluating whether a candidate eval should be added to a test suite.

CANDIDATE EVAL:
- User message: {user_message}
- Capability tested: {capability_tested}
- Cluster: {cluster_title}

EXISTING PATTERNS IN THE INDEX:
{existing_patterns}

Answer these questions:
1. Does this test something mechanically different from existing patterns? (Same primary behavior + same output type = same pattern, even if different business domain)
2. Would a regression here actually matter? (Is this non-trivial behavior worth protecting?)
3. Does this represent a distinct capability from existing evals in the same pattern?
4. Is the query clean and self-contained?

Also determine: if this matches an existing pattern, which one? If it's new, what should the pattern be named?

Respond in JSON:
{{
  "passes_gate": true or false,
  "reason": "1-2 sentence explanation",
  "matched_pattern": "name of existing pattern if matched, null if new",
  "suggested_pattern": "pattern name (for new patterns or to confirm match)",
  "is_duplicate_query": true or false,
  "capability_category": "short category label"
}}"""


def load_pattern_index(path: Path) -> PatternIndex:
    """Load pattern index from disk, or return empty."""
    if not path.exists():
        return PatternIndex()
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return PatternIndex(patterns=[PatternEntry(**p) for p in data])
    return PatternIndex(**data)


def save_pattern_index(index: PatternIndex, path: Path) -> None:
    """Persist pattern index to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"patterns": [p.model_dump() for p in index.patterns]},
        indent=2, ensure_ascii=False,
    ))


def run_coverage_gate(
    candidates: list[EvalItem],
    pattern_index: PatternIndex,
    llm: LLMClient,
    max_evals_per_pattern: int = 3,
) -> tuple[list[EvalItem], PatternIndex]:
    """Filter eval candidates through the coverage gate.

    Returns (accepted_evals, updated_pattern_index).
    """
    accepted = []

    for candidate in candidates:
        result = _evaluate_candidate(candidate, pattern_index, llm)
        if result is None:
            continue

        passes, matched_pattern, suggested_pattern, capability = result

        if not passes:
            continue

        if matched_pattern:
            # Known pattern — check cap
            entry = pattern_index.find_match(matched_pattern)
            if entry and entry.at_cap:
                continue
            if entry and _is_duplicate_query(candidate.user_message, entry.eval_queries):
                continue
            if entry:
                entry.evals_generated += 1
                entry.eval_queries.append(candidate.user_message)
                entry.has_eval = True
        else:
            # New pattern
            pattern_index.add_pattern(PatternEntry(
                pattern=suggested_pattern or candidate.capability_tested or candidate.cluster_title,
                description=candidate.judge_remarks,
                example=candidate.user_message,
                has_eval=True,
                evals_generated=1,
                eval_queries=[candidate.user_message],
                capabilities=[capability] if capability else [],
            ))

        accepted.append(candidate)

    return accepted, pattern_index


def _evaluate_candidate(
    candidate: EvalItem,
    pattern_index: PatternIndex,
    llm: LLMClient,
) -> Optional[tuple[bool, Optional[str], Optional[str], Optional[str]]]:
    """Evaluate a single candidate against the coverage gate.

    Returns (passes, matched_pattern_name, suggested_pattern_name, capability) or None on failure.
    """
    existing_summary = _format_patterns_for_prompt(pattern_index)

    prompt = COVERAGE_GATE_PROMPT.format(
        user_message=candidate.user_message,
        capability_tested=candidate.capability_tested or "unspecified",
        cluster_title=candidate.cluster_title,
        existing_patterns=existing_summary,
    )

    try:
        result = llm.chat_json(prompt)
        return (
            result.get("passes_gate", False),
            result.get("matched_pattern"),
            result.get("suggested_pattern"),
            result.get("capability_category"),
        )
    except Exception:
        # On LLM failure, use heuristic fallback
        return _heuristic_gate(candidate, pattern_index)


def _heuristic_gate(
    candidate: EvalItem,
    pattern_index: PatternIndex,
) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """Fallback: simple substring matching when LLM is unavailable."""
    capability = candidate.capability_tested or ""
    match = pattern_index.find_match(capability)

    if match:
        if match.at_cap:
            return (False, match.pattern, None, capability)
        if _is_duplicate_query(candidate.user_message, match.eval_queries):
            return (False, match.pattern, None, capability)
        return (True, match.pattern, None, capability)

    return (True, None, capability, capability)


def _is_duplicate_query(query: str, existing_queries: list[str]) -> bool:
    """Simple overlap check — if >60% of words match any existing query, it's a dup."""
    if not existing_queries:
        return False

    query_words = set(query.lower().split())
    if not query_words:
        return False

    for existing in existing_queries:
        existing_words = set(existing.lower().split())
        if not existing_words:
            continue
        overlap = len(query_words & existing_words) / max(len(query_words), len(existing_words))
        if overlap > 0.6:
            return True

    return False


def _format_patterns_for_prompt(index: PatternIndex) -> str:
    """Format pattern index as concise text for the LLM prompt."""
    if not index.patterns:
        return "(empty — no patterns yet)"

    lines = []
    for p in index.patterns:
        cap_status = "AT CAP" if p.at_cap else f"{p.evals_generated}/3"
        lines.append(f"- {p.pattern} [{cap_status}]: {p.description}")
        if p.eval_queries:
            for q in p.eval_queries[:2]:
                lines.append(f"    e.g. \"{q[:80]}\"")
    return "\n".join(lines)
