from pathlib import Path

import numpy as np

from open_standard_evaluation.checkpoint import CheckpointManager
from open_standard_evaluation.llm.client import LLMClient
from open_standard_evaluation.models.facet import Facet


def run_embedding(
    facets: list[Facet],
    llm: LLMClient,
    checkpoint: CheckpointManager,
) -> np.ndarray:
    """Generate embeddings for facets. Loads from cache if available."""
    embeddings_path = Path(checkpoint.cache_dir) / "embeddings.npy"

    existing = checkpoint.load_embeddings(len(facets), llm.embedding_model)

    if existing is not None:
        if existing.shape[0] == len(facets):
            print(f"  Loaded cached embeddings: {existing.shape}")
            return existing
        # Partial cache — embed only new facets
        if existing.shape[0] < len(facets):
            new_texts = [f.summary for f in facets[existing.shape[0]:]]
            print(f"  Embedding {len(new_texts)} new facets (appending to cache)...")
            new_embeddings = llm.embed(new_texts)
            all_embeddings = np.vstack([existing, new_embeddings])
            np.save(embeddings_path, all_embeddings)
            return all_embeddings

    # Full embedding run
    texts = [f.summary for f in facets]
    print(f"  Embedding {len(texts)} facets with {llm.embedding_model}...")
    embeddings = llm.embed(texts)
    np.save(embeddings_path, embeddings)

    meta = checkpoint.get_meta()
    meta["embedding_model"] = llm.embedding_model
    checkpoint.save_meta(meta)

    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings
