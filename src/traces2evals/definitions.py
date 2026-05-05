"""
User-provided metric definitions for enrichment.

Users place a `definitions.yaml` (or `definitions.md`) alongside their config.
This module loads those definitions and makes them available to the quality gate
and eval generation steps, so the pipeline understands platform-specific scoring.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class MetricDefinition(BaseModel):
    name: str
    scale: str  # e.g. "0-1", "1-5", "binary"
    description: str
    interpretation: dict[str, str] = {}
    reliability_notes: Optional[str] = None


class EnrichmentDefinitions(BaseModel):
    metrics: list[MetricDefinition] = []
    bucket_rules: dict[str, str] = {}
    noise_patterns: list[str] = []
    custom_instructions: Optional[str] = None

    @property
    def metric_names(self) -> list[str]:
        return [m.name for m in self.metrics]

    def get_metric(self, name: str) -> Optional[MetricDefinition]:
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def format_for_prompt(self) -> str:
        """Format definitions as context for LLM prompts."""
        if not self.metrics:
            return ""

        lines = ["Available production metrics:"]
        for m in self.metrics:
            lines.append(f"\n## {m.name}")
            lines.append(f"- Scale: {m.scale}")
            lines.append(f"- Measures: {m.description}")
            for key, val in m.interpretation.items():
                lines.append(f"- {key}: {val}")
            if m.reliability_notes:
                lines.append(f"- Note: {m.reliability_notes}")

        if self.custom_instructions:
            lines.append(f"\n## Additional context\n{self.custom_instructions}")

        return "\n".join(lines)


def load_definitions(path: Optional[Path] = None) -> EnrichmentDefinitions:
    """Load enrichment definitions from YAML file.

    Searches for: explicit path > definitions.yaml > definitions.yml in cwd.
    Returns empty definitions if no file found (graceful degradation).
    """
    candidates = [path] if path else [
        Path("definitions.yaml"),
        Path("definitions.yml"),
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            with open(candidate) as f:
                raw = yaml.safe_load(f) or {}
            return EnrichmentDefinitions(**raw)

    return EnrichmentDefinitions()
