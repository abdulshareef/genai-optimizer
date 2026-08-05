"""Model catalog and cost prediction.

Prices are USD per one million tokens, checked on 2026-08-05 from public
pricing pages. Vendors change prices often, so please treat this table as a
starting point and override it in your own project:

    from genai_optimizer.pricing import load_catalog
    load_catalog("my_prices.json")            # or set GENAI_OPTIMIZER_PRICING

``quality`` is a rough capability score out of 10, used only for model
selection. Tune it to match your own evaluation results.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

PRICES_UPDATED = "2026-08-05"


@dataclass
class ModelInfo:
    id: str
    provider: str
    input_price: float          # USD per 1M input tokens
    output_price: float         # USD per 1M output tokens
    context: int                # context window in tokens
    quality: float              # 0 to 10, capability
    speed: float                # 0 to 10, higher is faster
    cache_read_multiplier: float = 0.10
    batch_multiplier: float = 0.50
    tags: tuple = ()

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CATALOG: Dict[str, ModelInfo] = {m.id: m for m in [
    # ---- Anthropic -------------------------------------------------------
    ModelInfo("claude-fable-5", "anthropic", 10.0, 50.0, 1_000_000, 9.8, 5.0,
              tags=("frontier", "reasoning")),
    ModelInfo("claude-opus-5", "anthropic", 5.0, 25.0, 1_000_000, 9.5, 6.5,
              tags=("frontier", "coding", "agentic")),
    ModelInfo("claude-sonnet-5", "anthropic", 2.0, 10.0, 1_000_000, 9.0, 8.0,
              tags=("balanced", "coding")),
    ModelInfo("claude-haiku-4-5-20251001", "anthropic", 1.0, 5.0, 200_000, 7.5, 9.5,
              tags=("cheap", "fast")),
    # ---- OpenAI ----------------------------------------------------------
    ModelInfo("gpt-5.6-sol", "openai", 5.0, 30.0, 1_050_000, 9.5, 6.0,
              tags=("frontier",)),
    ModelInfo("gpt-5.6-terra", "openai", 2.0, 12.0, 1_050_000, 8.8, 8.0,
              tags=("balanced",)),
    ModelInfo("gpt-5.6-luna", "openai", 0.20, 1.20, 1_050_000, 7.2, 9.5,
              tags=("cheap", "fast")),
    # ---- Google ----------------------------------------------------------
    ModelInfo("gemini-3.1-pro", "google", 2.0, 12.0, 1_000_000, 9.0, 7.0,
              tags=("frontier", "long-context")),
    ModelInfo("gemini-3.6-flash", "google", 1.50, 7.50, 1_000_000, 8.5, 9.0,
              tags=("balanced", "fast")),
    ModelInfo("gemini-3.5-flash-lite", "google", 0.30, 2.50, 1_000_000, 7.0, 9.5,
              tags=("cheap", "fast")),
    ModelInfo("gemini-2.5-flash-lite", "google", 0.10, 0.40, 1_000_000, 6.0, 9.8,
              tags=("cheap", "fast", "legacy")),
    # ---- Local / self hosted, no per token cost --------------------------
    ModelInfo("llama3.1:8b", "ollama", 0.0, 0.0, 128_000, 6.0, 8.0,
              tags=("local", "free", "private")),
    ModelInfo("qwen2.5:14b", "ollama", 0.0, 0.0, 128_000, 6.8, 7.0,
              tags=("local", "free", "private")),
]}

CATALOG: Dict[str, ModelInfo] = dict(DEFAULT_CATALOG)


def load_catalog(path: Optional[str] = None, merge: bool = True) -> Dict[str, ModelInfo]:
    """Load model prices from a JSON file. Falls back to the built in table."""
    global CATALOG
    path = path or os.environ.get("GENAI_OPTIMIZER_PRICING")
    if not path:
        return CATALOG
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    loaded = {}
    for row in data.get("models", data if isinstance(data, list) else []):
        info = ModelInfo(
            id=row["id"], provider=row["provider"],
            input_price=float(row["input_price"]),
            output_price=float(row["output_price"]),
            context=int(row.get("context", 128000)),
            quality=float(row.get("quality", 7.0)),
            speed=float(row.get("speed", 7.0)),
            cache_read_multiplier=float(row.get("cache_read_multiplier", 0.10)),
            batch_multiplier=float(row.get("batch_multiplier", 0.50)),
            tags=tuple(row.get("tags", ())),
        )
        loaded[info.id] = info
    CATALOG = {**DEFAULT_CATALOG, **loaded} if merge else loaded
    return CATALOG


def reset_catalog() -> None:
    global CATALOG
    CATALOG = dict(DEFAULT_CATALOG)


def get_model(model_id: str) -> Optional[ModelInfo]:
    if model_id in CATALOG:
        return CATALOG[model_id]
    # allow short names like "claude-haiku-4-5" or "gpt-5.6"
    for key, info in CATALOG.items():
        if key.startswith(model_id) or model_id in key:
            return info
    return None


def list_models(provider: Optional[str] = None) -> List[ModelInfo]:
    models = list(CATALOG.values())
    if provider:
        models = [m for m in models if m.provider == provider]
    return sorted(models, key=lambda m: m.input_price)


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int,
                  cached_input_tokens: int = 0, batch: bool = False) -> float:
    """Return the estimated cost in USD. Unknown model gives 0.0."""
    info = get_model(model_id)
    if info is None:
        return 0.0
    fresh_input = max(0, input_tokens - cached_input_tokens)
    cost = (fresh_input / 1_000_000.0) * info.input_price
    cost += (cached_input_tokens / 1_000_000.0) * info.input_price * info.cache_read_multiplier
    cost += (output_tokens / 1_000_000.0) * info.output_price
    if batch:
        cost *= info.batch_multiplier
    return round(cost, 6)


def format_usd(amount: float) -> str:
    if amount == 0:
        return "$0"
    if amount < 0.01:
        return "$%.6f" % amount
    return "$%.4f" % amount
