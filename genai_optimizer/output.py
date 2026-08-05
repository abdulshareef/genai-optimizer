"""Output Optimizer.

Models pad their answers. This module removes the padding and the repeated
lines, but it will never drop a sentence that carries a fact which appears
nowhere else. After compressing, we verify every extracted fact is still
present. If a fact is missing, the sentence holding it is put back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

from .analyzer import Protector, split_sentences, _tidy
from .tokens import count_tokens, savings

# Openings and closings that carry no information
BOILERPLATE = [
    r"^(certainly|sure|of course|absolutely|great question|good question)[!,.]?\s*",
    r"^(as an? (ai|ai language model|assistant)[^.]*\.)\s*",
    r"\b(please note that)\b",
    r"\b(it'?s|it is) important to note that\b",
    r"\b(it should be noted that)\b",
    r"\b(it is worth (noting|mentioning) that)\b",
    r"\b(as mentioned (earlier|above|previously))\b",
    r"\b(in conclusion|to summarise|to summarize|in summary|to sum up|overall)[,:]?\s*",
    r"\b(remember|keep in mind) that\b",
    r"\b(needless to say|at the end of the day|when all is said and done)[,]?\s*",
]
_BOILER_RE = re.compile("|".join(BOILERPLATE), re.IGNORECASE)

# Whole sentences that exist only to be polite. These are dropped, not edited,
# so we never leave a broken half sentence behind.
FILLER_SENTENCES = [
    r"^(certainly|sure|absolutely|of course|great question|good question)[!,.]?$",
    r"^(i'?d|i would) be (very )?happy to help.{0,40}$",
    r"^(happy to help|let me help you|glad to help).{0,30}$",
    r"^i hope (this|that) (helps|is helpful).{0,40}$",
    r"^let me know if you (have|need|want).{0,60}$",
    r"^feel free to (ask|reach out|let me know).{0,60}$",
    r"^(here'?s|here is|below is) (a|an|the) [\w\s]{0,30}(overview|summary|breakdown|explanation|answer|list)[:.]?$",
    r"^as an? (ai|ai language model|assistant).{0,80}$",
    r"^(that'?s (it|all)|hope that (helps|clarifies)).{0,30}$",
]
_FILLER_SENT_RE = re.compile("|".join(FILLER_SENTENCES), re.IGNORECASE)


def _is_fragment(sentence: str) -> bool:
    words = [w for w in re.findall(r"[\w']+", sentence) if w]
    if not words:
        return True
    if len(words) <= 6 and re.match(
            r"^(in|for|to|at|of|with|so|and|but|as|the|a|an)\b", sentence.strip(), re.I):
        return True
    return False

# Anything matching these is a fact and must survive
FACT_PATTERNS = [
    r"\d+(?:[.,]\d+)*\s*%",                       # percentages
    r"(?:₹|Rs\.?|INR|\$|USD|€|£)\s*\d[\d,.]*",    # money
    r"\b\d{4}-\d{2}-\d{2}\b",                     # ISO dates
    r"\b\d+(?:[.,]\d+)?\s*(?:kg|g|mg|km|m|cm|mm|GB|MB|KB|TB|ms|s|min|hr|hours?|days?|weeks?|months?|years?)\b",
    r"\b\d+(?:[.,]\d+)?\b",                       # bare numbers
    r"https?://\S+",                              # links
    r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){1,3}\b",  # proper noun phrases
    r"\b(?:Section|Sec\.?|Rule|Clause|Article|CVE|ISO|IEC|RFC)\s*[\dA-Za-z().\-]+",
]
_FACT_RES = [re.compile(p) for p in FACT_PATTERNS]


@dataclass
class OutputResult:
    text: str
    original: str
    tokens_before: int = 0
    tokens_after: int = 0
    removed_sentences: int = 0
    facts_restored: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def percent_saved(self) -> float:
        return savings(self.tokens_before, self.tokens_after)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": max(0, self.tokens_before - self.tokens_after),
            "percent_saved": self.percent_saved,
            "removed_sentences": self.removed_sentences,
            "facts_restored": self.facts_restored,
            "notes": self.notes,
        }

    def summary(self) -> str:
        return "Output: %d -> %d tokens (%.1f%% saved), %d sentences removed" % (
            self.tokens_before, self.tokens_after, self.percent_saved,
            self.removed_sentences)


def extract_facts(text: str) -> Set[str]:
    found: Set[str] = set()
    for rx in _FACT_RES:
        for m in rx.finditer(text):
            token = m.group(0).strip()
            if len(token) > 1:
                found.add(token)
    return found


class OutputOptimizer:
    """Compress a model answer without losing information."""

    LEVELS = ("off", "safe", "balanced", "aggressive")

    def __init__(self, level: str = "balanced", preserve_facts: bool = True,
                 min_tokens: int = 40, target_ratio: float | None = None):
        if level not in self.LEVELS:
            raise ValueError("level must be one of %s" % (self.LEVELS,))
        self.level = level
        self.preserve_facts = preserve_facts
        self.min_tokens = min_tokens
        self.target_ratio = target_ratio   # e.g. 0.6 keeps about 60 percent

    def optimize(self, text: str) -> OutputResult:
        before = count_tokens(text)
        result = OutputResult(text=text, original=text, tokens_before=before,
                              tokens_after=before)
        if self.level == "off" or not text.strip():
            return result
        if before < self.min_tokens:
            result.notes.append("answer too short to compress safely")
            return result

        protector = Protector()
        working = protector.hide(text)

        original_facts = extract_facts(working) if self.preserve_facts else set()

        sentences = split_sentences(working)
        kept: List[str] = []
        seen: Set[str] = set()
        removed = 0

        for sentence in sentences:
            cleaned = sentence
            if self.level in ("balanced", "aggressive"):
                if _FILLER_SENT_RE.match(cleaned.strip().rstrip(".!")) or \
                        _FILLER_SENT_RE.match(cleaned.strip()):
                    removed += 1
                    continue
                cleaned = _BOILER_RE.sub(" ", cleaned)
            cleaned = _tidy(cleaned)
            if not cleaned or cleaned in ".,:;":
                removed += 1
                continue
            if self.level in ("balanced", "aggressive") and _is_fragment(cleaned) \
                    and not extract_facts(cleaned):
                removed += 1
                continue
            key = re.sub(r"[^a-z0-9]+", "", cleaned.lower())
            if key and key in seen:
                removed += 1              # exact repeat
                continue
            if self.level == "aggressive" and key and _near_duplicate(key, seen):
                removed += 1
                continue
            if key:
                seen.add(key)
            kept.append(cleaned)

        compressed = _join(kept)

        # Optional harder trim towards a size target, low value sentences first
        if self.target_ratio and 0 < self.target_ratio < 1:
            compressed, dropped = _trim_to_ratio(compressed, before, self.target_ratio)
            removed += dropped

        # Fact check, put back whatever went missing
        restored = 0
        if self.preserve_facts and original_facts:
            missing = {f for f in original_facts if f not in compressed}
            if missing:
                rescue = [s for s in sentences
                          if any(f in s for f in missing) and _tidy(s) not in compressed]
                if rescue:
                    compressed = _join([compressed] + [_tidy(s) for s in rescue])
                    restored = len(missing)
                    result.notes.append(
                        "%d fact(s) were about to be lost, sentences restored" % restored)

        final = protector.restore(compressed)
        result.text = final
        result.tokens_after = count_tokens(final)
        result.removed_sentences = removed
        result.facts_restored = restored

        # Never return something longer than what we started with
        if result.tokens_after >= result.tokens_before:
            result.text = text
            result.tokens_after = before
            result.notes.append("compression gave no benefit, original kept")
        return result


def _join(parts: List[str]) -> str:
    out = " ".join(p for p in parts if p)
    return re.sub(r"\s{2,}", " ", out).strip()


def _near_duplicate(key: str, seen: Set[str], threshold: float = 0.9) -> bool:
    for other in seen:
        if abs(len(other) - len(key)) > 12:
            continue
        shorter, longer = (key, other) if len(key) < len(other) else (other, key)
        if not longer:
            continue
        common = sum(1 for a, b in zip(shorter, longer) if a == b)
        if common / len(longer) >= threshold:
            return True
    return False


def _trim_to_ratio(text: str, original_tokens: int, ratio: float):
    """Drop the lowest value sentences until we reach the target size."""
    target = int(original_tokens * ratio)
    sentences = split_sentences(text)
    scored = []
    for i, s in enumerate(sentences):
        score = len(extract_facts(s)) * 3
        score += 2 if i == 0 else 0          # first sentence usually the answer
        score += 1 if re.match(r"^\s*[-*\d]", s) else 0
        scored.append((score, i, s))
    ordered = sorted(scored, key=lambda x: (x[0], -x[1]))
    drop = set()
    running = count_tokens(text)
    for score, i, s in ordered:
        if running <= target:
            break
        if score >= 3:
            continue                         # fact bearing, keep it
        drop.add(i)
        running -= count_tokens(s)
    kept = [s for i, s in enumerate(sentences) if i not in drop]
    return _join(kept), len(drop)
