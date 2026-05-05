from pathlib import Path
from typing import Optional

import click


@click.group()
def main():
    """traces2evals — Generate eval test suites from production agent traces."""
    pass


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
@click.option("--langfuse-host", default=None)
@click.option("--langfuse-public-key", default=None)
@click.option("--langfuse-secret-key", default=None)
@click.option("--model", default=None)
@click.option("--embedding-model", default=None)
@click.option("--output-format", type=click.Choice(["jsonl", "csv", "braintrust"]), default=None)
@click.option("--output-dir", default=None)
def run(config_path, langfuse_host, langfuse_public_key, langfuse_secret_key, model, embedding_model, output_format, output_dir):
    """Run the full pipeline end-to-end."""
    from traces2evals.config import load_config
    from traces2evals.pipeline import run_pipeline

    config = load_config(Path(config_path) if config_path else None)

    # CLI overrides
    if langfuse_host:
        config.source.langfuse.host = langfuse_host
    if langfuse_public_key:
        config.source.langfuse.public_key = langfuse_public_key
    if langfuse_secret_key:
        config.source.langfuse.secret_key = langfuse_secret_key
    if model:
        config.llm.model = model
    if embedding_model:
        config.llm.embedding_model = embedding_model
    if output_format:
        config.output.format = output_format
    if output_dir:
        config.output.directory = output_dir

    run_pipeline(config)


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
def status(config_path):
    """Show pipeline status and cached state."""
    from traces2evals.checkpoint import CheckpointManager
    from traces2evals.config import load_config

    config = load_config(Path(config_path) if config_path else None)
    cp = CheckpointManager(config.checkpoint.directory)

    print(f"Cache directory: {cp.cache_dir}")
    print(f"  Ingested:      {len(cp.get_completed_sessions('ingested'))}")
    print(f"  Quality-gated: {len(cp.get_completed_sessions('quality_gated'))}")
    print(f"  Faceted:       {len(cp.get_completed_sessions('faceted'))}")
    print(f"  Embedded:      {cp.artifact_exists('embeddings.npy')}")
    meta = cp.get_meta()
    if meta:
        print(f"  Embedding model: {meta.get('embedding_model', 'unknown')}")


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
def clean(config_path):
    """Remove all cached state and start fresh."""
    from traces2evals.checkpoint import CheckpointManager
    from traces2evals.config import load_config

    config = load_config(Path(config_path) if config_path else None)
    cp = CheckpointManager(config.checkpoint.directory)
    cp.clean()
    click.echo("Cache cleared.")


if __name__ == "__main__":
    main()
