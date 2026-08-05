"""Provider adapters.

Everything here uses only the Python standard library, so installing this
package does not drag in any SDK. If you already use the official SDK, you do
not need this at all. Just use ``engine.optimize_prompt`` and
``engine.optimize_output`` around your own calls.

Supported out of the box:
    anthropic, openai, google (gemini), ollama, and any OpenAI compatible
    endpoint (Groq, Together, OpenRouter, vLLM, LM Studio, Azure and so on).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


class ProviderError(RuntimeError):
    pass


def _post(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise ProviderError("HTTP %s from %s: %s" % (exc.code, url, detail)) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network
        raise ProviderError("Could not reach %s: %s" % (url, exc.reason)) from exc


class BaseProvider:
    name = "base"
    default_model = ""

    def complete(self, model: Optional[str], prompt: str, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.7
                 ) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    default_model = "claude-sonnet-5"

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.anthropic.com", **_):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    def complete(self, model, prompt, system=None, max_tokens=1024, temperature=0.7):
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        payload = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        data = _post(self.base_url + "/v1/messages", payload, {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        })
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, {"input_tokens": usage.get("input_tokens", 0),
                      "output_tokens": usage.get("output_tokens", 0)}


class OpenAIProvider(BaseProvider):
    name = "openai"
    default_model = "gpt-5.6-terra"

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", **_):
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.base_url = base_url.rstrip("/")

    def complete(self, model, prompt, system=None, max_tokens=1024, temperature=0.7):
        if not self.api_key:
            raise ProviderError("API key is not set for %s" % self.name)
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        payload = {"model": model or self.default_model, "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature}
        data = _post(self.base_url + "/chat/completions", payload,
                     {"Authorization": "Bearer " + self.api_key})
        text = data["choices"][0]["message"].get("content", "")
        usage = data.get("usage", {})
        return text, {"input_tokens": usage.get("prompt_tokens", 0),
                      "output_tokens": usage.get("completion_tokens", 0)}


class OpenAICompatibleProvider(OpenAIProvider):
    """For Groq, Together, OpenRouter, vLLM, LM Studio, Azure and friends."""
    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 api_key_env: str = "LLM_API_KEY", default_model: str = "", **_):
        super().__init__(api_key=api_key, base_url=base_url, api_key_env=api_key_env)
        if default_model:
            self.default_model = default_model


class GoogleProvider(BaseProvider):
    name = "google"
    default_model = "gemini-3.6-flash"

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta", **_):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") \
            or os.environ.get("GOOGLE_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    def complete(self, model, prompt, system=None, max_tokens=1024, temperature=0.7):
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is not set")
        model = model or self.default_model
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens,
                                 "temperature": temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = "%s/models/%s:generateContent?key=%s" % (self.base_url, model, self.api_key)
        data = _post(url, payload, {})
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata", {})
        return text, {"input_tokens": usage.get("promptTokenCount", 0),
                      "output_tokens": usage.get("candidatesTokenCount", 0)}


class OllamaProvider(BaseProvider):
    """Local models. No API key, no per token cost."""
    name = "ollama"
    default_model = "llama3.1:8b"

    def __init__(self, base_url: str = "http://localhost:11434", **_):
        self.base_url = base_url.rstrip("/")

    def complete(self, model, prompt, system=None, max_tokens=1024, temperature=0.7):
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        payload = {"model": model or self.default_model, "messages": messages,
                   "stream": False,
                   "options": {"temperature": temperature, "num_predict": max_tokens}}
        data = _post(self.base_url + "/api/chat", payload, {})
        text = data.get("message", {}).get("content", "")
        return text, {"input_tokens": data.get("prompt_eval_count", 0),
                      "output_tokens": data.get("eval_count", 0)}


REGISTRY = {
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,
    "openai": OpenAIProvider,
    "azure": OpenAIProvider,
    "google": GoogleProvider,
    "gemini": GoogleProvider,
    "ollama": OllamaProvider,
    "local": OllamaProvider,
    "compatible": OpenAICompatibleProvider,
}


def register_provider(name: str, cls) -> None:
    """Plug in your own provider class."""
    REGISTRY[name.lower()] = cls


def get_provider(name: str, **kwargs) -> BaseProvider:
    cls = REGISTRY.get((name or "").lower())
    if cls is None:
        raise ProviderError(
            "Unknown provider '%s'. Known: %s" % (name, ", ".join(sorted(REGISTRY))))
    return cls(**kwargs)


__all__ = ["BaseProvider", "AnthropicProvider", "OpenAIProvider",
           "OpenAICompatibleProvider", "GoogleProvider", "OllamaProvider",
           "ProviderError", "get_provider", "register_provider", "REGISTRY"]
