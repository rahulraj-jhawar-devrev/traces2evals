from typing import Optional

from pydantic import BaseModel


class PatternEntry(BaseModel):
    pattern: str
    description: str
    example: str
    has_eval: bool = False
    evals_generated: int = 0
    eval_queries: list[str] = []
    capabilities: list[str] = []

    @property
    def at_cap(self) -> bool:
        return self.evals_generated >= 3


class PatternIndex(BaseModel):
    patterns: list[PatternEntry] = []

    def find_match(self, capability: str, threshold: float = 0.8) -> Optional[PatternEntry]:
        """Find a pattern that matches the given capability (exact or substring)."""
        capability_lower = capability.lower()
        for p in self.patterns:
            if p.pattern.lower() == capability_lower:
                return p
            if capability_lower in p.pattern.lower() or p.pattern.lower() in capability_lower:
                return p
            for cap in p.capabilities:
                if cap.lower() == capability_lower:
                    return p
        return None

    def add_pattern(self, entry: PatternEntry) -> None:
        self.patterns.append(entry)

    def get_uncapped_patterns(self) -> list[PatternEntry]:
        return [p for p in self.patterns if not p.at_cap]

    def get_capped_patterns(self) -> list[PatternEntry]:
        return [p for p in self.patterns if p.at_cap]

    @property
    def total_evals(self) -> int:
        return sum(p.evals_generated for p in self.patterns)
