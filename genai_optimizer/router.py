"""Model selection.

Simple idea: work out the minimum capability the task needs, then pick the
cheapest model that clears that bar. This is where most of the real money is
saved, much more than prompt compression alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .analyzer import Analysis
from .pricing import ModelInfo, estimate_cost, list_models

# minimum quality score needed, by task type at complexity 0
TASK_QUALITY_FLOOR = {
    "extraction": 5.5, "chat": 5.5, "summarize": 6.0, "translation": 6.5,
    "creative": 6.8, "writing": 6.8, "code": 7.5, "reasoning": 7.8,
    "agentic": 8.0,
}

PRIORITIES = ("cost", "balanced", "quality", "speed")


@dataclass
class ModelChoice:
    model: str
    provider: str
    reason: str
    estimated_cost: float
    quality: float
    alternatives: List[dict]

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "reason": self.reason,
            "estimated_cost_usd": self.estimated_cost,
            "quality": self.quality,
            "alternatives": self.alternatives,
        }


def required_quality(analysis: Analysis) -> float:
    floor = TASK_QUALITY_FLOOR.get(analysis.task_type, 6.0)
    floor += analysis.complexity * 3.0
    if analysis.languages:
        floor += 0.3          # non English work needs a stronger model
    if analysis.has_code:
        floor += 0.2
    return round(min(9.9, floor), 2)


class ModelRouter:
    def __init__(self, priority: str = "balanced",
                 allowed_providers: Optional[List[str]] = None,
                 allowed_models: Optional[List[str]] = None,
                 pinned_model: Optional[str] = None,
                 include_local: bool = False):
        if priority not in PRIORITIES:
            raise ValueError("priority must be one of %s" % (PRIORITIES,))
        self.priority = priority
        self.allowed_providers = allowed_providers
        self.allowed_models = allowed_models
        self.pinned_model = pinned_model
        # Local models are free, so they would always win on cost. Keep them
        # out unless the caller asks for them.
        self.include_local = include_local or bool(
            allowed_providers and "ollama" in allowed_providers)

    def candidates(self) -> List[ModelInfo]:
        models = list_models()
        if self.allowed_providers:
            models = [m for m in models if m.provider in self.allowed_providers]
        if self.allowed_models:
            models = [m for m in models if m.id in self.allowed_models]
        if not self.include_local:
            models = [m for m in models if "local" not in m.tags]
        return models

    def select(self, analysis: Analysis, expected_output_tokens: int = 700) -> ModelChoice:
        need = required_quality(analysis)
        pool = self.candidates()
        if not pool:
            raise ValueError("No models available after filtering")

        fits = [m for m in pool if m.context >= analysis.input_tokens + expected_output_tokens]
        if not fits:
            fits = sorted(pool, key=lambda m: -m.context)[:1]

        good = [m for m in fits if m.quality >= need]
        if not good:
            # nothing clears the bar, take the strongest we have
            good = sorted(fits, key=lambda m: -m.quality)[:3]
            note = "no model met the required quality %.1f, using the strongest available" % need
        else:
            note = "needs quality >= %.1f for a %s task at complexity %.2f" % (
                need, analysis.task_type, analysis.complexity)

        def cost_of(m: ModelInfo) -> float:
            return estimate_cost(m.id, analysis.input_tokens, expected_output_tokens)

        if self.priority == "cost":
            ranked = sorted(good, key=lambda m: (cost_of(m), -m.quality))
        elif self.priority == "quality":
            ranked = sorted(good, key=lambda m: (-m.quality, cost_of(m)))
        elif self.priority == "speed":
            ranked = sorted(good, key=lambda m: (-m.speed, cost_of(m)))
        else:  # balanced, value for money
            ranked = sorted(good, key=lambda m: (cost_of(m) + 0.0001) / max(0.1, m.quality))

        if self.pinned_model:
            forced = [m for m in pool if m.id == self.pinned_model]
            if forced:
                ranked = forced + [m for m in ranked if m.id != self.pinned_model]
                note = "model pinned by configuration"

        best = ranked[0]
        alts = [{"model": m.id, "provider": m.provider,
                 "estimated_cost_usd": cost_of(m), "quality": m.quality}
                for m in ranked[1:4]]
        return ModelChoice(best.id, best.provider, note, cost_of(best), best.quality, alts)
