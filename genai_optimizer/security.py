"""Security validation for prompts and model output.

Three jobs:
  1. Catch prompt injection and jailbreak patterns.
  2. Catch secrets (API keys, tokens, private keys) before they leave your box.
  3. Catch PII, including Indian identifiers like Aadhaar, PAN and IFSC.

Modes:
  ``warn``   - only report (default)
  ``redact`` - mask the finding in the text
  ``block``  - raise SecurityBlocked when risk crosses the threshold
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


class SecurityBlocked(Exception):
    """Raised when mode is 'block' and the prompt crosses the risk threshold."""


@dataclass
class Finding:
    category: str          # injection | secret | pii | policy
    rule: str
    severity: str          # low | medium | high
    excerpt: str           # already masked, safe to log
    start: int = 0
    end: int = 0

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "rule": self.rule,
            "severity": self.severity,
            "excerpt": self.excerpt,
        }


@dataclass
class SecurityReport:
    findings: List[Finding] = field(default_factory=list)
    risk_score: int = 0            # 0 to 100
    blocked: bool = False
    text: str = ""                 # possibly redacted text

    @property
    def ok(self) -> bool:
        return not self.blocked and self.risk_score < 40

    def to_dict(self) -> dict:
        return {
            "risk_score": self.risk_score,
            "blocked": self.blocked,
            "findings": [f.to_dict() for f in self.findings],
        }


_SEVERITY_WEIGHT = {"low": 8, "medium": 20, "high": 40}

# --------------------------------------------------------------------------
# Rule tables
# --------------------------------------------------------------------------

INJECTION_RULES = [
    ("ignore_previous", r"\b(ignore|disregard|forget)\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instruction|prompt|rule|direction|message)", "high"),
    ("reveal_system", r"\b(reveal|show|print|repeat|output|dump)\s+(me\s+)?(your|the)\s+(system\s+prompt|initial\s+instruction|hidden\s+prompt|developer\s+message)", "high"),
    ("role_override", r"\byou\s+are\s+now\s+(a|an|the)?\s*\w+", "medium"),
    ("dev_mode", r"\b(developer\s+mode|dan\s+mode|jailbreak|god\s+mode|unrestricted\s+mode)\b", "high"),
    ("no_rules", r"\b(no|without)\s+(restrictions|filters|guardrails|safety|limitations)\b", "medium"),
    ("pretend", r"\b(pretend|act\s+as\s+if|simulate\s+that)\s+(you|there)\s+", "low"),
    ("exfiltrate", r"\b(send|post|upload|exfiltrate)\s+(this|the|all)\s+(data|conversation|context|file)s?\s+to\s+http", "high"),
    ("encoded_payload", r"\b(base64|rot13|hex)\s*(decode|decoded|encoded)\s*(this|the following)", "medium"),
    ("tool_abuse", r"\b(run|execute)\s+(this\s+)?(shell|bash|powershell|system)\s+command", "medium"),
]

SECRET_RULES = [
    ("openai_key", r"\bsk-[A-Za-z0-9_\-]{16,}\b", "high"),
    ("anthropic_key", r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b", "high"),
    ("google_key", r"\bAIza[0-9A-Za-z_\-]{30,}\b", "high"),
    ("aws_key", r"\b(AKIA|ASIA)[0-9A-Z]{16}\b", "high"),
    ("github_token", r"\b(ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{20,}\b", "high"),
    ("slack_token", r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b", "high"),
    ("private_key", r"-----BEGIN\s+(RSA|EC|OPENSSH|PGP|DSA)?\s*PRIVATE KEY-----", "high"),
    ("jwt", r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b", "medium"),
    ("password_assign", r"\b(password|passwd|pwd|secret|api[_\- ]?key)\s*[:=]\s*[^\s,;]{6,}", "medium"),
    ("conn_string", r"\b(mongodb|postgres|postgresql|mysql|redis)://[^\s]+:[^\s]+@[^\s]+", "high"),
]

PII_RULES = [
    ("email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "low"),
    ("phone_in", r"(?<!\d)(?:\+91[\-\s]?|0)?[6-9]\d{9}(?!\d)", "medium"),
    ("pan", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "high"),
    ("ifsc", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", "medium"),
    ("gstin", r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b", "medium"),
    ("ipv4", r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", "low"),
]

_AADHAAR_RE = re.compile(r"(?<!\d)[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)")

# Verhoeff tables, used to confirm an Aadhaar looking number is really valid.
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    digits = re.sub(r"\D", "", number)
    if len(digits) != 12:
        return False
    check = 0
    for i, ch in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[i % 8][int(ch)]]
    return check == 0


def luhn_valid(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def mask(value: str) -> str:
    """Mask a sensitive value so it is safe to write into a log or report."""
    value = value.strip()
    if len(value) <= 8:
        return value[0] + "*" * (len(value) - 1) if value else ""
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


class SecurityValidator:
    def __init__(self, mode: str = "warn", block_threshold: int = 60,
                 allow_pii: bool = False, custom_rules=None):
        if mode not in ("warn", "redact", "block"):
            raise ValueError("mode must be warn, redact or block")
        self.mode = mode
        self.block_threshold = block_threshold
        self.allow_pii = allow_pii
        self.custom_rules = list(custom_rules or [])   # (name, regex, severity)

    # ---------------------------------------------------------------- scan
    def scan(self, text: str) -> SecurityReport:
        findings: List[Finding] = []
        if not text:
            return SecurityReport(text=text)

        def collect(rules, category):
            for name, pattern, severity in rules:
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    findings.append(Finding(category, name, severity,
                                            mask(m.group(0)), m.start(), m.end()))

        collect(INJECTION_RULES, "injection")
        collect(SECRET_RULES, "secret")
        if not self.allow_pii:
            collect(PII_RULES, "pii")
        if self.custom_rules:
            collect(self.custom_rules, "policy")

        if not self.allow_pii:
            for m in _AADHAAR_RE.finditer(text):
                if verhoeff_valid(m.group(0)):
                    findings.append(Finding("pii", "aadhaar", "high",
                                            mask(m.group(0)), m.start(), m.end()))
            for m in _CARD_RE.finditer(text):
                if luhn_valid(m.group(0)):
                    findings.append(Finding("pii", "card_number", "high",
                                            mask(m.group(0)), m.start(), m.end()))

        score = 0
        for f in findings:
            score += _SEVERITY_WEIGHT[f.severity]
        score = min(100, score)

        out_text = text
        if self.mode == "redact" and findings:
            out_text = self._redact(text, findings)

        blocked = self.mode == "block" and score >= self.block_threshold
        report = SecurityReport(findings=findings, risk_score=score,
                                blocked=blocked, text=out_text)
        if blocked:
            raise SecurityBlocked(
                "Prompt blocked, risk score %d. Rules: %s"
                % (score, ", ".join(sorted({f.rule for f in findings})))
            )
        return report

    # -------------------------------------------------------------- redact
    def _redact(self, text: str, findings: List[Finding]) -> str:
        spans = sorted(
            [(f.start, f.end, f.rule) for f in findings if f.category != "injection"],
            key=lambda s: s[0], reverse=True,
        )
        for start, end, rule in spans:
            text = text[:start] + "[REDACTED:%s]" % rule.upper() + text[end:]
        return text
