"""Token counting.

We do not force any heavy dependency. If ``tiktoken`` is installed we use it,
otherwise we fall back to a heuristic counter that stays within roughly
10-15 percent of real tokenizers for normal English text.
"""

from __future__ import annotations

import math
import re

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

_ENCODER = None
_ENCODER_TRIED = False


def _get_encoder():
    global _ENCODER, _ENCODER_TRIED
    if _ENCODER_TRIED:
        return _ENCODER
    _ENCODER_TRIED = True
    try:  # pragma: no cover - depends on optional package
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER = None
    return _ENCODER


def heuristic_tokens(text: str) -> int:
    """Estimate tokens without any tokenizer library."""
    if not text:
        return 0
    total = 0
    for piece in _WORD_RE.findall(text):
        if piece.isascii():
            if piece.isalnum():
                # roughly 4 characters per token, minimum one token
                total += max(1, math.ceil(len(piece) / 4.0))
            else:
                total += 1
        else:
            # Indic, CJK and emoji text costs far more tokens per character
            total += max(1, math.ceil(len(piece) / 1.5))
    return total


def count_tokens(text: str, exact: bool = True) -> int:
    """Return the token count for ``text``.

    ``exact=True`` will use tiktoken when available. Set it to False if you
    want the fast heuristic every time.
    """
    if not text:
        return 0
    if exact:
        enc = _get_encoder()
        if enc is not None:  # pragma: no cover
            try:
                return len(enc.encode(text))
            except Exception:
                pass
    return heuristic_tokens(text)


def count_messages(messages, exact: bool = True) -> int:
    """Count tokens for a chat style message list, with per message overhead."""
    total = 0
    for msg in messages or []:
        content = msg.get("content", "")
        if isinstance(content, list):  # multimodal blocks
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += count_tokens(block.get("text", ""), exact)
                else:
                    total += 260  # rough placeholder for an image block
        else:
            total += count_tokens(str(content), exact)
        total += 4  # role and formatting overhead
    return total


def savings(before: int, after: int) -> float:
    """Percentage saved. Returns 0.0 when nothing was saved."""
    if before <= 0:
        return 0.0
    return round(max(0.0, (before - after) / before * 100.0), 2)
