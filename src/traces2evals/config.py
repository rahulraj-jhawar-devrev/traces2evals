import os
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class LangfuseSourceConfig(BaseModel):
    host: str = "https://cloud.langfuse.com"
    public_key: str = ""
    secret_key: str = ""
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    max_sessions: int = 5000
    min_turns: int = 2


class SourceConfig(BaseModel):
    adapter: str = "langfuse"
    langfuse: LangfuseSourceConfig = LangfuseSourceConfig()


class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.0
    max_parallel_calls: int = 5


class QualityConfig(BaseModel):
    use_platform_scores: bool = True
    score_name: Optional[str] = None
    min_quality_threshold: float = 0.6
    fallback_to_llm: bool = True
    exclude_failures: bool = False


class FacetsConfig(BaseModel):
    max_input_tokens: int = 10000
    batch_size: int = 20
    parallel_batch_size: int = 5


class ClusteringConfig(BaseModel):
    umap_n_components: int = 20
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    min_cluster_size: int = 5
    min_samples: int = 1
    cluster_selection_epsilon: float = 0.15
    cluster_selection_method: str = "eom"
    merge_similarity_threshold: float = 0.83
    min_cluster_pct: float = 0.1
    min_sessions_per_cluster: int = 2
    max_group_size: int = 10
    auto_tune: bool = False
    auto_tune_trials: int = 50


class EvalGenerationConfig(BaseModel):
    max_evals_per_cluster: int = 3
    synthesis_preference: str = "auto"
    privatize: bool = True


class OutputConfig(BaseModel):
    format: str = "jsonl"
    directory: str = "./output"


class CheckpointConfig(BaseModel):
    enabled: bool = True
    directory: str = "./.traces2evals_cache"


class TracesConfig(BaseModel):
    source: SourceConfig = SourceConfig()
    llm: LLMConfig = LLMConfig()
    quality: QualityConfig = QualityConfig()
    facets: FacetsConfig = FacetsConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    eval_generation: EvalGenerationConfig = EvalGenerationConfig()
    output: OutputConfig = OutputConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()


def _interpolate_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} with environment variable values."""
    pattern = re.compile(r"\$\{([^}]+)\}")
    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))
    return pattern.sub(replacer, value)


def _walk_and_interpolate(obj):
    """Recursively interpolate env vars in all string values."""
    if isinstance(obj, dict):
        return {k: _walk_and_interpolate(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_walk_and_interpolate(item) for item in obj]
    elif isinstance(obj, str):
        return _interpolate_env_vars(obj)
    return obj


def load_config(path: Optional[Path] = None) -> TracesConfig:
    """Load config from YAML file with env var interpolation."""
    if path is None:
        path = Path("traces2evals.yaml")

    if not path.exists():
        return TracesConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    interpolated = _walk_and_interpolate(raw)
    return TracesConfig(**interpolated)
