from pathlib import Path

from traces2evals.adapters.langfuse import LangfuseAdapter
from traces2evals.checkpoint import CheckpointManager
from traces2evals.config import TracesConfig
from traces2evals.llm.client import LLMClient
from traces2evals.output.jsonl import write_jsonl
from traces2evals.steps.cluster import run_clustering
from traces2evals.steps.embed import run_embedding
from traces2evals.steps.facets import run_facet_extraction
from traces2evals.steps.generate import run_eval_generation
from traces2evals.steps.ingest import run_ingest
from traces2evals.steps.label import run_labeling
from traces2evals.steps.quality_gate import run_quality_gate


def run_pipeline(config: TracesConfig) -> None:
    """Execute the full traces2evals pipeline end-to-end."""
    checkpoint = CheckpointManager(config.checkpoint.directory)
    llm = LLMClient(
        model=config.llm.model,
        embedding_model=config.llm.embedding_model,
        temperature=config.llm.temperature,
        max_parallel_calls=config.llm.max_parallel_calls,
    )

    # Step 1: Ingest
    print("=" * 60)
    print("STEP 1: Ingesting sessions from Langfuse")
    print("=" * 60)
    adapter = LangfuseAdapter(config.source.langfuse)
    adapter.health_check()
    sessions = run_ingest(adapter, config.source, checkpoint)
    print(f"  Sessions: {len(sessions)}")

    # Step 2: Quality gate
    print("\n" + "=" * 60)
    print("STEP 2: Quality gate")
    print("=" * 60)
    sessions = run_quality_gate(sessions, config.quality, llm, checkpoint)
    print(f"  Passed: {len(sessions)}")

    # Step 3: Facet extraction
    print("\n" + "=" * 60)
    print("STEP 3: Facet extraction")
    print("=" * 60)
    facets = run_facet_extraction(sessions, config.facets, llm, checkpoint)
    print(f"  Facets: {len(facets)}")

    # Step 4: Embedding
    print("\n" + "=" * 60)
    print("STEP 4: Embedding")
    print("=" * 60)
    embeddings = run_embedding(facets, llm, checkpoint)

    # Step 5: Clustering
    print("\n" + "=" * 60)
    print("STEP 5: Clustering (COMPASS)")
    print("=" * 60)
    labels, n_clusters, hierarchy = run_clustering(facets, embeddings, config.clustering)

    # Step 6: Labeling
    print("\n" + "=" * 60)
    print("STEP 6: Cluster labeling")
    print("=" * 60)
    cluster_hierarchy = run_labeling(facets, labels, n_clusters, hierarchy, llm)

    # Step 7: Eval generation
    print("\n" + "=" * 60)
    print("STEP 7: Eval generation")
    print("=" * 60)
    evals = run_eval_generation(sessions, facets, labels, cluster_hierarchy, config.eval_generation, llm)
    print(f"  Generated: {len(evals)} evals across {n_clusters} clusters")

    # Output
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output.format == "jsonl":
        write_jsonl(evals, output_dir / "evals.jsonl")
    elif config.output.format == "csv":
        from traces2evals.output.csv import write_csv
        write_csv(evals, output_dir / "evals.csv")
    elif config.output.format == "braintrust":
        from traces2evals.output.braintrust import write_braintrust
        write_braintrust(evals, output_dir / "evals.jsonl")

    print(f"\n{'=' * 60}")
    print(f"DONE — {len(evals)} evals written to {output_dir}/")
    print(f"{'=' * 60}")
