import json

import pytest

from open_standard_evaluation.checkpoint import CheckpointManager


class TestCheckpointManager:
    def test_init_creates_cache_dir(self, tmp_cache):
        cp = CheckpointManager(tmp_cache)
        assert cp.cache_dir.exists()

    def test_mark_and_get_completed(self, tmp_cache):
        cp = CheckpointManager(tmp_cache)
        cp.mark_completed("ingested", ["s1", "s2", "s3"])
        completed = cp.get_completed_sessions("ingested")
        assert completed == {"s1", "s2", "s3"}

    def test_get_completed_empty_step(self, tmp_cache):
        cp = CheckpointManager(tmp_cache)
        cp.mark_completed("ingested", ["s1"])
        assert cp.get_completed_sessions("faceted") == set()

    def test_multi_step_tracking(self, tmp_cache):
        cp = CheckpointManager(tmp_cache)
        cp.mark_completed("ingested", ["s1", "s2"])
        cp.mark_completed("quality_gated", ["s1"])
        assert cp.get_completed_sessions("ingested") == {"s1", "s2"}
        assert cp.get_completed_sessions("quality_gated") == {"s1"}

    def test_save_and_load_artifact(self, tmp_cache):
        cp = CheckpointManager(tmp_cache)
        data = {"scores": [0.8, 0.9, 0.7]}
        cp.save_artifact("quality_scores.json", data)
        loaded = cp.load_artifact("quality_scores.json")
        assert loaded == data

    def test_load_nonexistent_artifact(self, tmp_cache):
        cp = CheckpointManager(tmp_cache)
        assert cp.load_artifact("missing.json") is None

    def test_artifact_exists(self, tmp_cache):
        cp = CheckpointManager(tmp_cache)
        assert not cp.artifact_exists("test.json")
        cp.save_artifact("test.json", {"x": 1})
        assert cp.artifact_exists("test.json")

    def test_meta(self, tmp_cache):
        cp = CheckpointManager(tmp_cache)
        assert cp.get_meta() == {}
        cp.save_meta({"embedding_model": "text-embedding-3-large", "config_hash": "abc123"})
        meta = cp.get_meta()
        assert meta["embedding_model"] == "text-embedding-3-large"

    def test_clean(self, tmp_cache):
        cp = CheckpointManager(tmp_cache)
        cp.mark_completed("ingested", ["s1"])
        cp.save_artifact("data.json", {"x": 1})
        assert cp.cache_dir.exists()
        cp.clean()
        assert not cp.cache_dir.exists()

    def test_manifest_persists_across_instances(self, tmp_cache):
        cp1 = CheckpointManager(tmp_cache)
        cp1.mark_completed("ingested", ["s1", "s2"])
        cp2 = CheckpointManager(tmp_cache)
        assert cp2.get_completed_sessions("ingested") == {"s1", "s2"}
