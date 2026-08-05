"""The Optimization Engine.

    User Prompt -> Intent Analyzer -> LLM -> Output Optimizer

The engine is provider neutral. You can use only the optimizer part and keep
your existing SDK, or let the engine call the provider for you.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .analyzer import Analysis, IntentAnalyzer
from .output import OutputOptimizer, OutputResult
from .pricing import estimate_cost, format_usd, get_model
from .router import ModelChoice, ModelRouter
from .security import SecurityReport, SecurityValidator
from .tokens import count_messages, count_tokens, savings


@dataclass
class PromptResult:
    text: str
    original: str
    analysis: Analysis
    security: SecurityReport
    model_choice: Optional[ModelChoice] = None
    tokens_before: int = 0
    tokens_after: int = 0
    estimated_cost_before: float = 0.0
    estimated_cost_after: float = 0.0
    elapsed_ms: float = 0.0

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def percent_saved(self) -> float:
        return savings(self.tokens_before, self.tokens_after)

    @property
    def cost_saved(self) -> float:
        return round(max(0.0, self.estimated_cost_before - self.estimated_cost_after), 6)

    def to_dict(self) -> dict:
        return {
            "optimized_prompt": self.text,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "percent_saved": self.percent_saved,
            "estimated_cost_before_usd": self.estimated_cost_before,
            "estimated_cost_after_usd": self.estimated_cost_after,
            "cost_saved_usd": self.cost_saved,
            "analysis": self.analysis.to_dict(),
            "security": self.security.to_dict(),
            "model_choice": self.model_choice.to_dict() if self.model_choice else None,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }

    def summary(self) -> str:
        lines = [
            "Task            : %s (complexity %.2f)" % (self.analysis.task_type,
                                                        self.analysis.complexity),
            "Prompt tokens   : %d -> %d  (%.1f%% saved)" % (
                self.tokens_before, self.tokens_after, self.percent_saved),
        ]
        if self.model_choice:
            lines.append("Model selected  : %s [%s] - %s" % (
                self.model_choice.model, self.model_choice.provider,
                self.model_choice.reason))
        lines.append("Estimated cost  : %s -> %s (saved %s)" % (
            format_usd(self.estimated_cost_before),
            format_usd(self.estimated_cost_after),
            format_usd(self.cost_saved)))
        lines.append("Security        : risk %d/100, %d finding(s)" % (
            self.security.risk_score, len(self.security.findings)))
        return "\n".join(lines)


@dataclass
class RunResult:
    prompt: PromptResult
    answer: str
    raw_answer: str
    output: OutputResult
    usage: Dict[str, Any] = field(default_factory=dict)
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "model": self.model,
            "usage": self.usage,
            "prompt": self.prompt.to_dict(),
            "output": self.output.to_dict(),
        }

    def summary(self) -> str:
        return self.prompt.summary() + "\n" + self.output.summary()


class OptimizationEngine:
    """One object for the whole pipeline.

    >>> engine = OptimizationEngine()
    >>> res = engine.optimize_prompt("Could you please kindly write a short note")
    >>> res.text
    'Write a short note'
    """

    def __init__(self,
                 level: str = "balanced",
                 output_level: str = "balanced",
                 security_mode: str = "warn",
                 priority: str = "balanced",
                 allowed_providers: Optional[List[str]] = None,
                 allowed_models: Optional[List[str]] = None,
                 pinned_model: Optional[str] = None,
                 include_local: bool = False,
                 preserve_facts: bool = True,
                 select_model: bool = True,
                 expected_output_tokens: int = 700,
                 pricing_file: Optional[str] = None):
        if pricing_file:
            from .pricing import load_catalog
            load_catalog(pricing_file)
        self.analyzer = IntentAnalyzer(level=level)
        self.output_optimizer = OutputOptimizer(level=output_level,
                                                preserve_facts=preserve_facts)
        self.validator = SecurityValidator(mode=security_mode)
        self.router = ModelRouter(priority=priority,
                                  allowed_providers=allowed_providers,
                                  allowed_models=allowed_models,
                                  pinned_model=pinned_model,
                                  include_local=include_local)
        self.select_model_enabled = select_model
        self.expected_output_tokens = expected_output_tokens
        self.stats = {"calls": 0, "tokens_saved": 0, "cost_saved_usd": 0.0}

    # ------------------------------------------------------------- prompts
    def optimize_prompt(self, prompt: str, system: Optional[str] = None,
                        model: Optional[str] = None) -> PromptResult:
        start = time.perf_counter()
        joined = (system + "\n\n" + prompt) if system else prompt

        security = self.validator.scan(joined)
        source = security.text if self.validator.mode == "redact" else joined

        analysis = self.analyzer.analyse(source)
        compressed = self.analyzer.compress(source, analysis)

        before = count_tokens(joined)
        after = count_tokens(compressed)

        choice = None
        if self.select_model_enabled and model is None:
            choice = self.router.select(analysis, self.expected_output_tokens)
            model_for_cost = choice.model
        else:
            model_for_cost = model or "claude-sonnet-5"

        cost_before = estimate_cost(model_for_cost, before, self.expected_output_tokens)
        cost_after = estimate_cost(model_for_cost, after, self.expected_output_tokens)

        result = PromptResult(
            text=compressed, original=prompt, analysis=analysis, security=security,
            model_choice=choice, tokens_before=before, tokens_after=after,
            estimated_cost_before=cost_before, estimated_cost_after=cost_after,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
        self.stats["calls"] += 1
        self.stats["tokens_saved"] += result.tokens_saved
        self.stats["cost_saved_usd"] = round(
            self.stats["cost_saved_usd"] + result.cost_saved, 6)
        return result

    def optimize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop in helper. Give it an OpenAI or Anthropic style message list,
        get back the same list with each text content compressed."""
        out = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                new = dict(msg)
                new["content"] = self.analyzer.compress(content)
                out.append(new)
            elif isinstance(content, list):
                blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        block = dict(block)
                        block["text"] = self.analyzer.compress(block.get("text", ""))
                    blocks.append(block)
                new = dict(msg)
                new["content"] = blocks
                out.append(new)
            else:
                out.append(msg)
        return out

    # -------------------------------------------------------------- output
    def optimize_output(self, text: str) -> OutputResult:
        return self.output_optimizer.optimize(text)

    # ----------------------------------------------------------- full trip
    def run(self, prompt: str, system: Optional[str] = None,
            provider: Optional[str] = None, model: Optional[str] = None,
            max_tokens: int = 1024, temperature: float = 0.7,
            optimize_answer: bool = True, **kwargs) -> RunResult:
        """Optimize, call the provider, then optimize the answer."""
        from .providers import get_provider

        pr = self.optimize_prompt(prompt, system=system, model=model)

        chosen_model = model or (pr.model_choice.model if pr.model_choice else None)
        chosen_provider = provider or (pr.model_choice.provider if pr.model_choice else None)
        if chosen_provider is None:
            info = get_model(chosen_model or "")
            chosen_provider = info.provider if info else "openai"

        client = get_provider(chosen_provider, **kwargs)
        raw, usage = client.complete(
            model=chosen_model, prompt=pr.text, system=system,
            max_tokens=max_tokens, temperature=temperature)

        out = (self.optimize_output(raw) if optimize_answer
               else self.output_optimizer.__class__(level="off").optimize(raw))

        if usage:
            actual = estimate_cost(chosen_model or "",
                                   usage.get("input_tokens", 0),
                                   usage.get("output_tokens", 0))
            usage["actual_cost_usd"] = actual

        return RunResult(prompt=pr, answer=out.text, raw_answer=raw,
                         output=out, usage=usage or {}, model=chosen_model or "")

    # --------------------------------------------------------------- misc
    def report(self) -> dict:
        return dict(self.stats)


# Convenience one liners --------------------------------------------------

_default_engine: Optional[OptimizationEngine] = None


def _engine() -> OptimizationEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = OptimizationEngine()
    return _default_engine


def optimize(prompt: str, **kwargs) -> PromptResult:
    """Quick helper: ``from genai_optimizer import optimize``."""
    return _engine().optimize_prompt(prompt, **kwargs)


def optimize_output(text: str) -> OutputResult:
    return _engine().optimize_output(text)
