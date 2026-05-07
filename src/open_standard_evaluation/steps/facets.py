import tiktoken
from tqdm import tqdm

from open_standard_evaluation.checkpoint import CheckpointManager
from open_standard_evaluation.config import FacetsConfig
from open_standard_evaluation.llm.client import LLMClient
from open_standard_evaluation.models.facet import Facet
from open_standard_evaluation.models.session import NormalizedSession
from open_standard_evaluation.prompts import FACET_EXTRACTION_PROMPT


def run_facet_extraction(
    sessions: list[NormalizedSession],
    config: FacetsConfig,
    llm: LLMClient,
    checkpoint: CheckpointManager,
) -> list[Facet]:
    """Extract privatized facets from sessions via LLM."""
    already_done = checkpoint.get_completed_sessions("faceted")
    to_process = [s for s in sessions if s.session_id not in already_done]

    # Load existing facets from cache
    existing = checkpoint.load_artifact("facets.json") or []
    facets = [Facet(**f) for f in existing]

    if not to_process:
        return facets

    print(f"  Extracting facets for {len(to_process)} sessions...")

    batch_results = []
    for i in tqdm(range(0, len(to_process), config.batch_size), desc="Facet extraction"):
        batch = to_process[i : i + config.batch_size]
        prompts = [
            FACET_EXTRACTION_PROMPT.format(
                conversation_text=_truncate(s.conversation_text, config.max_input_tokens),
                num_turns=s.num_turns,
            )
            for s in batch
        ]

        results = llm.chat_json_batch(prompts, desc="")
        for session, result in zip(batch, results):
            facet_list = result.get("facets", [])
            if not facet_list and "summary" in result:
                facet_list = [{"summary": result["summary"], "num_turns": session.num_turns}]
            if not facet_list:
                facet_list = [{"summary": session.user_messages_text[:200], "num_turns": session.num_turns}]

            for idx, f in enumerate(facet_list):
                batch_results.append(Facet(
                    session_id=session.session_id,
                    facet_idx=idx,
                    summary=f.get("summary", ""),
                    num_turns=f.get("num_turns", session.num_turns),
                    user_id=session.user_id,
                    user_text=session.user_messages_text[:500],
                ))

        checkpoint.mark_completed("faceted", [s.session_id for s in batch])

    facets.extend(batch_results)
    checkpoint.save_artifact("facets.json", [f.model_dump() for f in facets])
    return facets


def _truncate(text: str, max_tokens: int) -> str:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return text[:max_tokens * 4]
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])
