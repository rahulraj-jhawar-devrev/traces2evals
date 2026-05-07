from pathlib import Path

from open_standard_evaluation.adapters.langfuse import LangfuseAdapter
from open_standard_evaluation.checkpoint import CheckpointManager
from open_standard_evaluation.config import TracesConfig
from open_standard_evaluation.definitions import load_definitions
from open_standard_evaluation.llm.client import LLMClient
from open_standard_evaluation.output.jsonl import write_jsonl
from open_standard_evaluation.steps.cluster import run_clustering
from open_standard_evaluation.steps.coverage_gate import load_pattern_index, run_coverage_gate, save_pattern_index
from open_standard_evaluation.steps.embed import run_embedding
from open_standard_evaluation.steps.facets import run_facet_extraction
from open_standard_evaluation.steps.generate import run_eval_generation
from open_standard_evaluation.steps.ingest import run_ingest
from open_standard_evaluation.steps.label import run_labeling
from open_standard_evaluation.steps.quality_gate import run_quality_gate
from open_standard_evaluation.steps.sanitize import sanitize_eval_items


def run_pipeline(config: TracesConfig) -> None:
    """Execute the full open_standard_evaluation pipeline end-to-end."""
    checkpoint = CheckpointManager(config.checkpoint.directory)
    llm = LLMClient(
        model=config.llm.model,
        embedding_model=config.llm.embedding_model,
        temperature=config.llm.temperature,
        max_parallel_calls=config.llm.max_parallel_calls,
    )

    # Load enrichment definitions (optional)
    definitions_path = Path(config.definitions_path) if config.definitions_path else None
    definitions = load_definitions(definitions_path)
    if definitions.metrics:
        print(f"Loaded definitions: {', '.join(definitions.metric_names)}")

    # Step 1: Ingest
    print("=" * 60)
    print("STEP 1: Ingesting sessions from Langfuse")
    print("=" * 60)
    adapter = LangfuseAdapter(config.source.langfuse)
    adapter.health_check()
    sessions = run_ingest(adapter, config.source, checkpoint)
    print(f"  Sessions: {len(sessions)}")

    # Step 2: Quality gate (definitions-aware)
    print("\n" + "=" * 60)
    print("STEP 2: Quality gate")
    print("=" * 60)
    sessions = run_quality_gate(sessions, config.quality, llm, checkpoint, definitions)
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
    print(f"  Candidates: {len(evals)} evals across {n_clusters} clusters")

    # Step 8: Coverage gate (dedup + pattern tracking)
    if config.eval_generation.coverage_gate:
        print("\n" + "=" * 60)
        print("STEP 8: Coverage gate")
        print("=" * 60)
        pattern_path = Path(config.eval_generation.pattern_index_path)
        pattern_index = load_pattern_index(pattern_path)
        print(f"  Loaded pattern index: {len(pattern_index.patterns)} existing patterns")

        evals, pattern_index = run_coverage_gate(
            evals, pattern_index, llm, config.eval_generation.max_evals_per_pattern
        )
        save_pattern_index(pattern_index, pattern_path)
        print(f"  Accepted: {len(evals)} evals ({len(pattern_index.patterns)} total patterns)")

    # Step 9: Sanitize reference responses
    print("\n" + "=" * 60)
    print("STEP 9: Sanitize responses")
    print("=" * 60)
    evals = sanitize_eval_items(evals)
    print(f"  Sanitized: {len(evals)} evals")

    # Output
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output.format == "jsonl":
        write_jsonl(evals, output_dir / "evals.jsonl")
    elif config.output.format == "csv":
        from open_standard_evaluation.output.csv import write_csv
        write_csv(evals, output_dir / "evals.csv")
    elif config.output.format == "braintrust":
        from open_standard_evaluation.output.braintrust import write_braintrust
        write_braintrust(evals, output_dir / "evals.jsonl")

    print(f"\n{'=' * 60}")
    print(f"DONE — {len(evals)} evals written to {output_dir}/")
    if config.eval_generation.coverage_gate:
        print(f"Pattern index: {len(pattern_index.patterns)} patterns tracked")
    print(f"{'=' * 60}")
