import numpy as np
from tqdm import tqdm

from traces2evals.config import EvalGenerationConfig
from traces2evals.llm.client import LLMClient
from traces2evals.models.cluster import ClusterHierarchy
from traces2evals.models.eval_item import EvalItem
from traces2evals.models.facet import Facet
from traces2evals.models.session import NormalizedSession
from traces2evals.prompts import EVAL_CRAFTING_PROMPT


def run_eval_generation(
    sessions: list[NormalizedSession],
    facets: list[Facet],
    labels: np.ndarray,
    cluster_hierarchy: ClusterHierarchy,
    config: EvalGenerationConfig,
    llm: LLMClient,
) -> list[EvalItem]:
    """Generate eval items — up to max_evals_per_cluster per cluster."""
    session_map = {s.session_id: s for s in sessions}
    evals = []

    for cluster in tqdm(cluster_hierarchy.clusters, desc="Generating evals"):
        cluster_facets = [
            f for i, f in enumerate(facets) if labels[i] == cluster.cluster_id
        ]
        if not cluster_facets:
            continue

        # Select diverse sessions (pick facets with most distinct user_text)
        selected = _select_diverse(cluster_facets, config.max_evals_per_cluster)

        for facet in selected:
            session = session_map.get(facet.session_id)
            if not session:
                continue

            prompt = EVAL_CRAFTING_PROMPT.format(
                cluster_title=cluster.title,
                cluster_description=cluster.description,
                conversation_text=session.conversation_text[:6000],
            )

            try:
                result = llm.chat_json(prompt)
                evals.append(EvalItem(
                    user_message=result.get("user_message", ""),
                    reference_response=result.get("reference_response", ""),
                    judge_remarks=result.get("judge_remarks", ""),
                    cluster_id=cluster.cluster_id,
                    cluster_title=cluster.title,
                    source_session_id=facet.session_id,
                    synthesis_type=result.get("synthesis_type", "verbatim"),
                    capability_tested=result.get("capability_tested"),
                ))
            except Exception as e:
                print(f"  Failed to generate eval for session {facet.session_id}: {e}")
                continue

    return evals


def _select_diverse(facets: list[Facet], max_count: int) -> list[Facet]:
    """Select diverse facets from a cluster — prefer different sessions and intents."""
    seen_sessions = set()
    selected = []
    for f in facets:
        if f.session_id in seen_sessions:
            continue
        seen_sessions.add(f.session_id)
        selected.append(f)
        if len(selected) >= max_count:
            break
    return selected
