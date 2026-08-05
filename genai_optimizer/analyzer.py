"""Intent Analyzer.

Reads the user prompt and works out:
  * what kind of task it is,
  * how hard it is,
  * which parts are hard constraints that must never be touched,
  * which parts are filler that can safely go away.

Compression is rule based and deterministic. No model call, no network, so it
adds well under a millisecond for a normal prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

# --------------------------------------------------------------------------
# Task detection
# --------------------------------------------------------------------------

TASK_KEYWORDS: Dict[str, List[str]] = {
    "code": ["code", "function", "class", "bug", "debug", "refactor", "python",
             "javascript", "sql", "api", "compile", "stack trace", "unit test",
             "regex", "script", "docker", "typescript", "java", "c#", "rust"],
    "reasoning": ["prove", "why", "explain how", "derive", "analyse", "analyze",
                  "step by step", "logic", "calculate", "solve", "theorem",
                  "trade-off", "root cause", "compare and decide"],
    "extraction": ["extract", "parse", "list all", "pull out", "find all",
                   "classify", "categorise", "categorize", "label", "tag",
                   "json output", "structured output", "table of"],
    "summarize": ["summarise", "summarize", "summary", "tl;dr", "key points",
                  "condense", "brief me", "abstract"],
    "creative": ["story", "poem", "script", "screenplay", "song", "creative",
                 "imagine", "character", "novel", "dialogue", "tagline"],
    "translation": ["translate", "translation", "in hindi", "in malayalam",
                    "in tamil", "in kannada", "localise", "localize"],
    "agentic": ["tool", "agent", "workflow", "browse", "search the web",
                "call the api", "multi-step", "orchestrate"],
    "writing": ["report", "draft", "email", "letter", "proposal", "article",
                "blog post", "documentation", "sop", "policy", "essay",
                "write a note", "press release", "minutes"],
}

# how much brain power each task type usually needs, 0 to 1
TASK_BASE_COMPLEXITY = {
    "code": 0.62, "reasoning": 0.70, "agentic": 0.72, "creative": 0.45,
    "summarize": 0.28, "extraction": 0.22, "translation": 0.30, "chat": 0.20,
    "writing": 0.42,
}

# --------------------------------------------------------------------------
# Constraint detection - these sentences are protected during compression
# --------------------------------------------------------------------------

CONSTRAINT_PATTERNS = [
    r"\bmust\b", r"\bmust not\b", r"\bshould not\b", r"\bdo not\b", r"\bdon't\b",
    r"\bnever\b", r"\balways\b", r"\bonly\b", r"\brequired\b", r"\bmandatory\b",
    r"\bexactly\b", r"\bat least\b", r"\bat most\b", r"\bno more than\b",
    r"\bformat\b", r"\bjson\b", r"\bcsv\b", r"\bmarkdown\b", r"\byaml\b",
    r"\bschema\b", r"\bwords?\b\s*(limit|max|maximum)", r"\bin\s+\d+\s+words\b",
    r"\bbullet points?\b", r"\btable\b", r"\bdeadline\b", r"\bversion\b",
    r"\breturn only\b", r"\boutput only\b", r"\brespond in\b", r"\bwrite in\b",
    r"\bunder\s+\d+", r"\b\d+\s*(words|lines|rows|items|points|steps)\b",
]
_CONSTRAINT_RE = re.compile("|".join(CONSTRAINT_PATTERNS), re.IGNORECASE)

# --------------------------------------------------------------------------
# Filler that adds tokens but no meaning
# --------------------------------------------------------------------------

FILLER_PHRASES = [
    r"\bplease\b", r"\bkindly\b", r"\bif you (could|can|would|don't mind)\b",
    r"\bi (would|'d) (like|want) you to\b", r"\bi want you to\b",
    r"\bi was (wondering|hoping) if\b", r"\bcould you (please\s+)?",
    r"\bcan you (please\s+)?", r"\bwould you (please\s+)?",
    r"\bi need you to\b", r"\bhelp me to\b",
    r"\bas an? (ai|assistant|language model)\b",
    r"\bfor me\b", r"\bthank you( so much)?\b", r"\bthanks( a lot| in advance)?\b",
    r"\bbasically\b", r"\bactually\b", r"\breally\b", r"\bvery much\b",
    r"\bjust\b", r"\bsort of\b", r"\bkind of\b", r"\ba bit\b",
    r"\bin my opinion\b", r"\bi think that\b", r"\bit seems like\b",
    r"\bthat being said\b", r"\bneedless to say\b", r"\bas you know\b",
    r"\bat the end of the day\b", r"\bwhen it comes to\b",
    r"\bfeel free to\b", r"\bdo let me know\b", r"\bhope (this|that) helps\b",
]
_FILLER_RE = re.compile("|".join(FILLER_PHRASES), re.IGNORECASE)

# verbose phrase -> short phrase
PHRASE_MAP = [
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bfor the reason that\b", "because"),
    (r"\bin the event that\b", "if"),
    (r"\bat this point in time\b", "now"),
    (r"\bat the present moment\b", "now"),
    (r"\bin the near future\b", "soon"),
    (r"\ba large number of\b", "many"),
    (r"\ba small number of\b", "few"),
    (r"\bthe majority of\b", "most"),
    (r"\bis able to\b", "can"),
    (r"\bare able to\b", "can"),
    (r"\bhas the ability to\b", "can"),
    (r"\bit is important to note that\b", ""),
    (r"\bit should be noted that\b", ""),
    (r"\bit is worth mentioning that\b", ""),
    (r"\bplease be advised that\b", ""),
    (r"\bwith regard to\b", "about"),
    (r"\bwith reference to\b", "about"),
    (r"\bin relation to\b", "about"),
    (r"\bin terms of\b", "for"),
    (r"\bprior to\b", "before"),
    (r"\bsubsequent to\b", "after"),
    (r"\bin spite of the fact that\b", "although"),
    (r"\bon a regular basis\b", "regularly"),
    (r"\bmake use of\b", "use"),
    (r"\bcarry out\b", "do"),
    (r"\bgive an explanation of\b", "explain"),
    (r"\bprovide a description of\b", "describe"),
    (r"\bconduct an analysis of\b", "analyse"),
    (r"\bin a detailed manner\b", "in detail"),
    (r"\bcomprehensive and detailed\b", "detailed"),
    (r"\bcompletely and totally\b", "fully"),
    (r"\beach and every\b", "every"),
    (r"\bfirst and foremost\b", "first"),
    (r"\bvarious different\b", "various"),
    (r"\babsolutely essential\b", "essential"),
    (r"\bend result\b", "result"),
    (r"\bfuture plans\b", "plans"),
    (r"\bpast history\b", "history"),
    (r"\bnew innovation\b", "innovation"),
    (r"\bfree gift\b", "gift"),
]

# whole sentences that carry no instruction at all
DROP_SENTENCES = [
    r"^(hi|hello|hey|dear|good (morning|evening|afternoon))\b.{0,40}$",
    r"^i hope (you|this) (are|is|find)\b.*$",
    r"^(thanks|thank you|many thanks|appreciate it)\b.{0,40}$",
    r"^(you could |can you |could you )?help me (out )?(with (this|it|something))?[.!]?$",
    r"^i (have|had|got) a (small |quick )?(question|request|doubt)[.!]?$",
    r"^(quick|small) question[.!]?$",
    r"^i need (some |your )?help[.!]?$",
    r"^(in|for|to|at|of|with|so|and|but|as)\b[\w\s]{0,25}[.!]?$",
    r"^(that'?s (it|all)|that is all)[.!]?$",
]
_DROP_RE = re.compile("|".join(DROP_SENTENCES), re.IGNORECASE)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Analysis:
    task_type: str = "chat"
    complexity: float = 0.3
    constraints: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    needs_long_context: bool = False
    input_tokens: int = 0
    question_count: int = 0
    has_code: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "complexity": round(self.complexity, 2),
            "constraints": self.constraints,
            "needs_long_context": self.needs_long_context,
            "input_tokens": self.input_tokens,
            "has_code": self.has_code,
            "notes": self.notes,
        }


class Protector:
    """Hides code blocks, quoted text and <keep> blocks during compression."""

    PATTERNS = [
        re.compile(r"```.*?```", re.DOTALL),
        re.compile(r"<keep>.*?</keep>", re.DOTALL | re.IGNORECASE),
        re.compile(r"`[^`\n]+`"),
        re.compile(r"\{\{.*?\}\}", re.DOTALL),
        re.compile(r"https?://\S+"),
    ]

    def __init__(self):
        self.store: List[str] = []

    def hide(self, text: str) -> str:
        for pattern in self.PATTERNS:
            def repl(m):
                self.store.append(m.group(0))
                return "\x00P%dP\x00" % (len(self.store) - 1)
            text = pattern.sub(repl, text)
        return text

    def restore(self, text: str) -> str:
        for i, original in enumerate(self.store):
            text = text.replace("\x00P%dP\x00" % i, original)
        return text


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def is_constraint(sentence: str) -> bool:
    return bool(_CONSTRAINT_RE.search(sentence))


class IntentAnalyzer:
    """Analyse and compress a prompt without losing the actual instruction."""

    LEVELS = ("off", "safe", "balanced", "aggressive")

    def __init__(self, level: str = "balanced", min_tokens: int = 12,
                 keep_constraints: bool = True):
        if level not in self.LEVELS:
            raise ValueError("level must be one of %s" % (self.LEVELS,))
        self.level = level
        self.min_tokens = min_tokens
        self.keep_constraints = keep_constraints

    # ------------------------------------------------------------- analyse
    def analyse(self, prompt: str) -> Analysis:
        low = prompt.lower()
        scores = {}
        for task, words in TASK_KEYWORDS.items():
            scores[task] = sum(1 for w in words if w in low)
        task = max(scores, key=scores.get) if any(scores.values()) else "chat"

        has_code = "```" in prompt or bool(re.search(r"\b(def |class |import |SELECT |function\()", prompt))
        if has_code and scores.get("code", 0) == 0:
            task = "code"

        from .tokens import count_tokens
        tok = count_tokens(prompt)

        questions = prompt.count("?")
        constraints = [s for s in split_sentences(prompt) if is_constraint(s)]

        complexity = TASK_BASE_COMPLEXITY.get(task, 0.3)
        complexity += min(0.20, tok / 6000.0)         # long prompt, more work
        complexity += min(0.10, questions * 0.025)    # many questions
        complexity += min(0.10, len(constraints) * 0.02)
        if has_code:
            complexity += 0.05
        if re.search(r"\b(production|court|legal|medical|financial|audit|forensic)\b", low):
            complexity += 0.08
        complexity = round(min(1.0, complexity), 3)

        langs = []
        for name, pattern in (("hi", r"[\u0900-\u097F]"), ("ml", r"[\u0D00-\u0D7F]"),
                              ("ta", r"[\u0B80-\u0BFF]"), ("kn", r"[\u0C80-\u0CFF]"),
                              ("te", r"[\u0C00-\u0C7F]"), ("ar", r"[\u0600-\u06FF]"),
                              ("zh", r"[\u4E00-\u9FFF]")):
            if re.search(pattern, prompt):
                langs.append(name)

        return Analysis(
            task_type=task, complexity=complexity, constraints=constraints,
            languages=langs, needs_long_context=tok > 60000, input_tokens=tok,
            question_count=questions, has_code=has_code,
        )

    # ------------------------------------------------------------ compress
    def compress(self, prompt: str, analysis: Analysis | None = None) -> str:
        if self.level == "off" or not prompt.strip():
            return prompt
        analysis = analysis or self.analyse(prompt)
        if analysis.input_tokens < self.min_tokens:
            return prompt  # too short, not worth the risk

        protector = Protector()
        text = protector.hide(prompt)

        # Keep constraint sentences aside so nothing important is lost.
        sentences = split_sentences(text)
        protected_idx = {i for i, s in enumerate(sentences) if is_constraint(s)} \
            if self.keep_constraints else set()

        out: List[str] = []
        seen = set()
        for i, sentence in enumerate(sentences):
            cleaned = self._compress_sentence(sentence, protected=i in protected_idx)
            if not cleaned:
                continue
            fingerprint = re.sub(r"[^a-z0-9]+", "", cleaned.lower())
            if fingerprint and fingerprint in seen:
                continue          # duplicate sentence, drop it
            if fingerprint:
                seen.add(fingerprint)
            out.append(cleaned)

        text = " ".join(out)
        text = _tidy(text)

        # Final safety net. If any constraint signature went missing, put the
        # original constraint sentence back, word for word.
        for i in sorted(protected_idx):
            original = sentences[i]
            if not _signature_present(original, text):
                text = text + " " + _tidy(original)
        text = _tidy(text)
        text = protector.restore(text)

        # Safety net. If we somehow ate the whole prompt, return the original.
        if not text.strip():
            return prompt
        return text

    def _compress_sentence(self, sentence: str, protected: bool = False) -> str:
        """Compress one sentence.

        A protected sentence carries a constraint. It still loses filler like
        'please' and 'kindly', because that never changes the instruction, but
        it is never dropped and never gets the aggressive clause surgery.
        """
        s = _apply_phrase_map(sentence)
        if self.level in ("balanced", "aggressive"):
            if not protected and _DROP_RE.match(s.strip()):
                return ""
            s = _FILLER_RE.sub(" ", s)
            s = _tidy(s)
            if not protected:
                if _DROP_RE.match(s.strip()) or _is_fragment(s):
                    return ""
        if self.level == "aggressive" and not protected:
            s = re.sub(r"\b(that|which|who)\s+(is|are|was|were)\b", "", s, flags=re.I)
            s = re.sub(r"\bthe\s+(?=[a-z]+\s+(of|in|for|to)\b)", "", s, flags=re.I)
        s = _tidy(s)
        if protected and not _signature_present(sentence, s):
            return _tidy(sentence)      # compression broke it, keep original
        return s


def constraint_signature(sentence: str) -> List[str]:
    """The bits of a constraint that must survive: numbers and format words."""
    signature = re.findall(r"\d+(?:[.,]\d+)*", sentence)
    signature += re.findall(
        r"\b(json|csv|xml|yaml|markdown|html|table|bullet points?|schema|"
        r"words?|lines?|rows|items|points|steps|paragraphs?)\b",
        sentence, re.IGNORECASE)
    return [s.lower() for s in signature]


def _signature_present(original: str, compressed: str) -> bool:
    low = compressed.lower()
    return all(token in low for token in constraint_signature(original))


def _apply_phrase_map(text: str) -> str:
    for pattern, replacement in PHRASE_MAP:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _is_fragment(sentence: str) -> bool:
    """True when filler removal left behind a meaningless stub."""
    words = [w for w in re.findall(r"[\w']+", sentence) if w]
    if not words:
        return True
    if len(words) <= 6 and re.match(
            r"^(in|for|to|at|of|with|so|and|but|as|the|a|an)\b", sentence.strip(), re.I):
        return True
    return False


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])\1+", r"\1", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip(" ,;")
    if text and text[0].islower() and len(text) > 1:
        text = text[0].upper() + text[1:]
    return text.strip()
