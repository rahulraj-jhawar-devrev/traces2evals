import json
from pathlib import Path
from typing import Any, Optional


class CheckpointManager:
    """Manages per-step checkpoints for resumable pipeline runs."""

    def __init__(self, cache_dir: str = "./.traces2evals_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self._manifest: Optional[dict] = None

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            if self.manifest_path.exists():
                self._manifest = json.loads(self.manifest_path.read_text())
            else:
                self._manifest = {}
        return self._manifest

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def get_completed_sessions(self, step: str) -> set[str]:
        """Return session IDs that have completed a given step."""
        return {
            sid for sid, steps in self.manifest.items()
            if isinstance(steps, dict) and steps.get(step)
        }

    def mark_completed(self, step: str, session_ids: list[str]) -> None:
        """Mark sessions as completed for a step."""
        for sid in session_ids:
            if sid not in self.manifest:
                self.manifest[sid] = {}
            self.manifest[sid][step] = True
        self._save_manifest()

    def save_artifact(self, name: str, data: Any) -> Path:
        """Save a JSON-serializable artifact to the cache."""
        path = self.cache_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".json") or name.endswith(".jsonl"):
            path.write_text(json.dumps(data, indent=2, default=str))
        else:
            path.write_text(str(data))
        return path

    def load_artifact(self, name: str) -> Optional[Any]:
        """Load a previously saved artifact."""
        path = self.cache_dir / name
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def artifact_exists(self, name: str) -> bool:
        return (self.cache_dir / name).exists()

    def get_meta(self) -> dict:
        """Load run metadata (embedding model, config hash, etc.)."""
        return self.load_artifact("meta.json") or {}

    def save_meta(self, meta: dict) -> None:
        self.save_artifact("meta.json", meta)

    def clean(self) -> None:
        """Remove all cached state."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self._manifest = None
