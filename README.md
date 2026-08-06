# genai-optimizer

A small optimisation layer that sits around **any** Gen AI model and cuts your
token bill, without losing your instructions or the facts in the answer.

```
User Prompt
    |
    v
Intent Analyzer
    |-- Remove redundancy
    |-- Preserve constraints
    |-- Compress semantically
    |-- Estimate token savings
    |-- Predict cost
    |-- Select best model
    |-- Security validation
    |
    v
  LLM  (Claude, GPT, Gemini, Llama, or anything else)
    |
    v
Output Optimizer
    |-- Remove repetition
    |-- Compress response
    |-- Preserve facts
    |-- Show token savings
```

## Why this exists

Most teams pay for three things they never wanted: polite filler in prompts,
repeated sentences in answers, and a frontier model doing a job that a cheap
model could have done. This library handles all three.

Typical result on real prompts: **30 to 50 percent fewer input tokens**,
**35 to 60 percent fewer output tokens**, and a model choice that often costs
5 to 20 times less for simple tasks.

## Main points

- **Works with every provider.** Anthropic, OpenAI, Google Gemini, Ollama,
  Groq, Together, OpenRouter, vLLM, Azure, or your own endpoint.
- **Zero dependencies.** Pure standard library. `pip install genai-optimizer`
  and nothing else comes along. `tiktoken` is used only if you already have it.
- **Nothing is lost.** Constraints (word limits, JSON, format, must and must
  not) are protected. Code blocks, links and `<keep>` blocks are never touched.
  Every fact in an answer is verified after compression, and restored if it
  went missing.
- **Deterministic and fast.** Rule based, no extra model call, under a
  millisecond for a normal prompt. Nothing leaves your machine.
- **Security built in.** Prompt injection, secrets and PII, including Aadhaar
  (Verhoeff checked), PAN, IFSC, GSTIN and card numbers (Luhn checked).
- **Use it your way.** Python library, CLI, or a small HTTP service for your
  PHP, Node, Java or n8n stack.

## Install

```bash
pip install genai-optimizer
# or straight from source
git clone https://github.com/abdulshareef/genai-optimizer
cd genai-optimizer && pip install -e .
```

## Quick start

```python
from genai_optimizer import OptimizationEngine

engine = OptimizationEngine()

result = engine.optimize_prompt(
    "Hi, I hope you are doing well. I was wondering if you could please "
    "kindly write a detailed incident response report for a phishing case. "
    "It must be under 800 words and must include a timeline table. Thanks!"
)

print(result.text)
# Write a detailed incident response report for a phishing case.
# It must be under 800 words and must include a timeline table.

print(result.summary())
# Task            : writing (complexity 0.48)
# Prompt tokens   : 137 -> 70  (48.9% saved)
# Model selected  : gemini-3.6-flash [google] - needs quality >= 8.3 ...
# Estimated cost  : $0.005456 -> $0.005355 (saved $0.000101)
# Security        : risk 0/100, 0 finding(s)
```

Notice that the word limit and the table requirement survived. That is the
whole point.

## Integration, three ways

### 1. Keep your own SDK (recommended)

Add two lines around the call you already have. Nothing else changes.

```python
from genai_optimizer import OptimizationEngine
import anthropic

engine = OptimizationEngine()
client = anthropic.Anthropic()

opt = engine.optimize_prompt(user_prompt)          # before

response = client.messages.create(
    model=opt.model_choice.model,                  # or your own fixed model
    max_tokens=1024,
    messages=[{"role": "user", "content": opt.text}],
)

answer = engine.optimize_output(response.content[0].text)   # after
print(answer.text, answer.summary())
```

Same idea with OpenAI, using the message helper:

```python
messages = engine.optimize_messages([
    {"role": "system", "content": "You are a careful and helpful assistant."},
    {"role": "user", "content": user_prompt},
])
completion = openai_client.chat.completions.create(model="gpt-5.6-terra",
                                                   messages=messages)
```

### 2. Let the engine make the call

```python
run = engine.run("Summarise this SOC alert in five bullet points: ...",
                 provider="anthropic")     # or openai, google, ollama
print(run.answer)
print(run.summary())
```

Set the key in your environment first: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
or `GEMINI_API_KEY`. Ollama needs no key.

### 3. Not using Python? Run the service

```bash
genai-optimize serve --port 8088
```

```bash
curl -s localhost:8088/optimize \
     -H 'Content-Type: application/json' \
     -d '{"prompt":"Could you please kindly summarise this for me, thanks","level":"balanced"}'
```

```json
{
  "optimized_prompt": "Summarise this",
  "tokens_before": 13,
  "tokens_after": 4,
  "percent_saved": 69.23,
  "model_choice": { "model": "gemini-2.5-flash-lite", "provider": "google" },
  "security": { "risk_score": 0, "findings": [] }
}
```

A ready made JavaScript client is in `clients/optimizer.js`.

## Command line

```bash
genai-optimize prompt "your long prompt here"
genai-optimize prompt --file brief.txt --level aggressive --json
genai-optimize output --file answer.txt
genai-optimize models --provider anthropic
```

## Settings

```python
engine = OptimizationEngine(
    level="balanced",          # off | safe | balanced | aggressive
    output_level="balanced",
    security_mode="warn",      # warn | redact | block
    priority="balanced",       # cost | balanced | quality | speed
    allowed_providers=["anthropic"],   # stay with one vendor
    pinned_model=None,         # force one model, skip routing
    include_local=False,       # set True to allow free Ollama models
    preserve_facts=True,
)
```

**Compression levels**

| Level | What it does | Use when |
|---|---|---|
| `off` | nothing | debugging, or a golden prompt you never want changed |
| `safe` | only verbose phrase rewrites | legal, medical, court filings |
| `balanced` | phrases, filler, greetings, duplicates | default for most work |
| `aggressive` | also drops low value clauses | high volume, cost sensitive jobs |

Anything inside triple backticks, single backticks, `{{template}}` braces,
URLs or `<keep>...</keep>` is never modified at any level.

## Security validation

```python
from genai_optimizer import SecurityValidator

report = SecurityValidator(mode="redact").scan(user_text)
print(report.risk_score)          # 0 to 100
print(report.text)                # secrets replaced with [REDACTED:...]
for f in report.findings:
    print(f.category, f.rule, f.severity, f.excerpt)   # excerpt is masked
```

Use `mode="block"` in production to raise `SecurityBlocked` when the risk
crosses your threshold. Findings never contain the raw secret, so they are
safe to write into your logs.

## Model prices

The built in catalog was checked on **2026-08-05**. Vendors change prices
often, so please keep your own copy:

```python
from genai_optimizer.pricing import load_catalog
load_catalog("my_prices.json")     # or set GENAI_OPTIMIZER_PRICING
```

```json
{"models": [
  {"id": "my-model", "provider": "openai", "input_price": 1.0,
   "output_price": 4.0, "context": 200000, "quality": 8.0, "speed": 8.0}
]}
```

The `quality` score is only used for routing. Tune it with your own evaluation
numbers, do not trust the defaults blindly.

## What it does not do

- It is not a semantic rewriter. It will not turn a bad prompt into a good one.
- It cannot guarantee identical model output after compression. Test on your
  own workload before turning on `aggressive`.
- The heuristic token counter is close, not exact. Install `tiktoken` if you
  need tighter numbers, and always trust the provider usage figures for billing.
- The security scanner is a first line of defence, not a complete DLP product.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## Licence

MIT. Built by [TALFOR Cybersecurity & Digital Forensics](https://talfor.in),
Bengaluru.
