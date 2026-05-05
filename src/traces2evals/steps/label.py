import numpy as np
from tqdm import tqdm

from traces2evals.config import ClusteringConfig
from traces2evals.llm.client import LLMClient
from traces2evals.models.cluster import ClusterHierarchy, ClusterInfo
from traces2evals.models.facet import Facet
from traces2evals.prompts import CLUSTER_LABELING_PROMPT, PARENT_LABELING_PROMPT


def run_labeling(
    facets: list[Facet],
    labels: np.ndarray,
    n_clusters: int,
    hierarchy: list[dict[int, int]],
    llm: LLMClient,
) -> ClusterHierarchy:
    """Generate LLM labels for all clusters and hierarchy levels."""
    clusters = []

    for cid in tqdm(range(n_clusters), desc="Labeling clusters"):
        mask = labels == cid
        cluster_facets = [facets[i] for i in range(len(facets)) if mask[i]]
        if not cluster_facets:
            continue

        samples = cluster_facets[:5]
        samples_formatted = "\n".join(
            f"  {i+1}. {f.user_text[:300] if f.user_text else f.summary}"
            for i, f in enumerate(samples)
        )

        prompt = CLUSTER_LABELING_PROMPT.format(samples_formatted=samples_formatted)
        try:
            result = llm.chat_json(prompt)
        except Exception:
            result = {"title": f"Cluster {cid}", "description": "Unable to generate label"}

        session_ids = {f.session_id for f in cluster_facets}
        user_ids = {f.user_id for f in cluster_facets if f.user_id}

        clusters.append(ClusterInfo(
            cluster_id=cid,
            title=result.get("title", f"Cluster {cid}"),
            description=result.get("description", ""),
            session_count=len(session_ids),
            user_count=len(user_ids) if user_ids else None,
        ))

    return ClusterHierarchy(clusters=clusters, levels=hierarchy)
