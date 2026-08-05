"""genai-optimizer

A small, provider neutral optimisation layer for any Gen AI model.

    User Prompt -> Intent Analyzer -> LLM -> Output Optimizer

Quick start:

    from genai_optimizer import OptimizationEngine

    engine = OptimizationEngine()
    result = engine.optimize_prompt("Could you kindly please write a short note")
    print(result.text)
    print(result.summary())

Built by TALFOR Cybersecurity & Digital Forensics. MIT licensed.
"""

from .analyzer import Analysis, IntentAnalyzer
from .engine import (OptimizationEngine, PromptResult, RunResult, optimize,
                     optimize_output)
from .output import OutputOptimizer, OutputResult
from .pricing import (CATALOG, ModelInfo, estimate_cost, get_model,
                      list_models, load_catalog)
from .router import ModelChoice, ModelRouter
from .security import (Finding, SecurityBlocked, SecurityReport,
                       SecurityValidator)
from .tokens import count_messages, count_tokens

__version__ = "1.0.0"
__author__ = "Dr. Abdul Shareef Pallivalappil, TALFOR Cybersecurity & Digital Forensics"

__all__ = [
    "OptimizationEngine", "PromptResult", "RunResult",
    "IntentAnalyzer", "Analysis",
    "OutputOptimizer", "OutputResult",
    "ModelRouter", "ModelChoice",
    "SecurityValidator", "SecurityReport", "SecurityBlocked", "Finding",
    "count_tokens", "count_messages",
    "estimate_cost", "get_model", "list_models", "load_catalog", "CATALOG",
    "ModelInfo", "optimize", "optimize_output", "__version__",
]
