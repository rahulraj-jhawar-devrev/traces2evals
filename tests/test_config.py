import os
from pathlib import Path

import pytest
import yaml

from open_standard_evaluation.config import (
    TracesConfig,
    _interpolate_env_vars,
    _walk_and_interpolate,
    load_config,
)


class TestEnvVarInterpolation:
    def test_interpolates_existing_var(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "secret123")
        assert _interpolate_env_vars("${TEST_KEY}") == "secret123"

    def test_leaves_missing_var_unchanged(self):
        result = _interpolate_env_vars("${DEFINITELY_NOT_SET_XYZ}")
        assert result == "${DEFINITELY_NOT_SET_XYZ}"

    def test_interpolates_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        result = _interpolate_env_vars("${HOST}:${PORT}")
        assert result == "localhost:8080"

    def test_walk_handles_nested_dicts(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "value")
        data = {"a": {"b": "${MY_VAR}"}, "c": [1, "${MY_VAR}"]}
        result = _walk_and_interpolate(data)
        assert result == {"a": {"b": "value"}, "c": [1, "value"]}


class TestLoadConfig:
    def test_loads_defaults_when_no_file(self, tmp_path):
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.llm.model == "gpt-4o-mini"
        assert config.source.adapter == "langfuse"
        assert config.clustering.min_cluster_size == 5

    def test_loads_from_yaml(self, tmp_path):
        cfg = {
            "llm": {"model": "anthropic/claude-sonnet-4-20250514", "temperature": 0.3},
            "output": {"format": "csv"},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg))

        config = load_config(path)
        assert config.llm.model == "anthropic/claude-sonnet-4-20250514"
        assert config.llm.temperature == 0.3
        assert config.output.format == "csv"
        # Defaults preserved
        assert config.clustering.umap_n_components == 20

    def test_interpolates_env_vars_in_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LF_KEY", "pk-test")
        cfg = {"source": {"langfuse": {"public_key": "${LF_KEY}"}}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg))

        config = load_config(path)
        assert config.source.langfuse.public_key == "pk-test"


class TestConfigDefaults:
    def test_all_defaults_valid(self):
        config = TracesConfig()
        assert config.source.langfuse.max_sessions == 5000
        assert config.quality.min_quality_threshold == 0.6
        assert config.facets.batch_size == 20
        assert config.eval_generation.max_evals_per_cluster == 3
        assert config.checkpoint.enabled is True
