"""Utilities for dynamic EOS handling and stop decisions.

These helpers approximate how the original Chatterbox service scheduled its
stop token logic while giving us more visibility into the stop decision.  The
goal is to dynamically compute per-prompt length expectations and combine
those expectations with token-level log probability signals so that we can
decide when to terminate generation without relying on brittle global
thresholds.

The logic operates in three stages:

1.  :func:`analyze_prompt_window` inspects the normalized text prompt to build
    a :class:`PromptWindow` describing the expected token span.
2.  :func:`apply_dynamic_stop` takes the raw token stream (and optional
    log-probability traces) and determines where to truncate.
3.  Heuristics such as repetition detection and guard caps ensure that we bail
    out from degenerate loops even when the model fails to emit an explicit
    stop token.

The heuristics are intentionally conservative for short prompts and gradually
become stricter as we observe stronger evidence (via log-probs or token
patterns) that a completion should end.  This mirrors the behaviour of the
"old" Chatterbox stack that shipped dynamic EOS logic without per-language
hand-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Optional, Sequence


# --- Prompt length analysis -------------------------------------------------


_BRACKET_CLEAN_RE = re.compile(r"\[[^\]]+\]")


def _is_cjk(char: str) -> bool:
    """Return True if the character is in a CJK block."""

    if not char:
        return False
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Unified Ideographs Extension A
        or 0x20000 <= code <= 0x2A6DF  # Extension B
        or 0x2A700 <= code <= 0x2B73F  # Extension C
        or 0x2B740 <= code <= 0x2B81F  # Extension D
        or 0x2B820 <= code <= 0x2CEAF  # Extension E
        or 0x2CEB0 <= code <= 0x2EBEF  # Extension F
        or 0x3000 <= code <= 0x303F  # CJK punctuation
        or 0x3040 <= code <= 0x30FF  # Hiragana/Katakana
        or 0x31F0 <= code <= 0x31FF  # Katakana Phonetic Extensions
    )


def _is_hangul(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return (0xAC00 <= code <= 0xD7A3) or (0x1100 <= code <= 0x11FF)


def _count_word_like_units(text: str) -> tuple[int, int, int, int]:
    """Return counts for (word-like units, cjk chars, digits, punctuation)."""

    word_like = 0
    in_word = False
    cjk = 0
    digits = 0
    punctuation = 0

    for ch in text:
        if ch.isspace():
            in_word = False
            continue

        category = unicodedata.category(ch)

        if _is_cjk(ch) or _is_hangul(ch):
            # Treat each syllable/character as its own unit; this better matches
            # the old production heuristics for CJK languages where whitespace is
            # sparse or absent.
            word_like += 1
            cjk += 1
            in_word = False
            continue

        if category.startswith("N"):
            digits += 1

        if category.startswith("P"):
            punctuation += 1

        if category.startswith(("L", "N")):
            if not in_word:
                word_like += 1
                in_word = True
        else:
            in_word = False

    return word_like, cjk, digits, punctuation


@dataclass
class PromptWindow:
    clean_text: str
    char_len: int
    word_like: int
    cjk_like: int
    digit_count: int
    punctuation_count: int
    estimated_tokens: int
    min_tokens: int
    guard_tokens: int
    model_cap: int
    stop_confidence: float
    debug: dict[str, Any] = field(default_factory=dict)


def analyze_prompt_window(
    prompt: str,
    *,
    temperature: float,
    max_model_len: int,
    max_tokens_limit: int,
    language_hint: Optional[str] = None,
) -> PromptWindow:
    """Derive a dynamic token window for the given prompt.

    The heuristic mirrors the legacy Chatterbox service: we derive an expected
    completion length from the prompt's grapheme statistics, clamp it to the
    request/model limits, and compute guard rails (``min_tokens`` and
    ``guard_tokens``) to avoid both premature and runaway generations.
    """

    language_hint = (language_hint or "").lower()

    clean = _BRACKET_CLEAN_RE.sub("", prompt or "").strip()
    char_len = sum(1 for ch in clean if not ch.isspace())

    if char_len == 0:
        return PromptWindow(
            clean_text="",
            char_len=0,
            word_like=0,
            cjk_like=0,
            digit_count=0,
            punctuation_count=0,
            estimated_tokens=0,
            min_tokens=0,
            guard_tokens=0,
            model_cap=min(max_tokens_limit, max_model_len),
            stop_confidence=0.75,
            debug={"reason": "empty"},
        )

    word_like, cjk_like, digit_count, punctuation_count = _count_word_like_units(clean)
    if word_like == 0:
        word_like = char_len

    avg_chars_per_word = char_len / max(word_like, 1)
    cjk_ratio = cjk_like / max(char_len, 1)

    # Base tokens-per-char using a smooth logistic-style curve.
    base_tpc = 1.65 + 0.55 * math.tanh((avg_chars_per_word - 2.0) / 2.4)

    # Adjust for predominantly CJK content (shorter units) or very long words.
    if cjk_ratio > 0.35 or language_hint in {"zh", "ja", "ko"}:
        base_tpc = min(base_tpc, 1.2 + 0.5 * cjk_ratio + 0.1 * avg_chars_per_word)
    elif language_hint in {"de", "fi", "sv"}:
        base_tpc += 0.2

    # Digits and punctuation tend to expand token counts slightly.
    base_tpc += min(0.3, 0.08 * punctuation_count)
    base_tpc += min(0.2, 0.05 * digit_count)

    # Clamp to a sensible range observed in the legacy system.
    base_tpc = max(1.05, min(2.85, base_tpc))

    estimated_tokens = max(8, int(round(char_len * base_tpc)))

    # Compute a minimum token allowance: short prompts get a fixed floor, longer
    # prompts depend on the estimated length and current sampling temperature.
    temperature = float(max(0.0, min(temperature, 1.5)))
    min_tokens = int(round(estimated_tokens * (0.45 + 0.18 * (1.0 - temperature))))
    min_tokens = max(12, min_tokens)

    # Guard tokens define a soft upper bound where we start forcing termination.
    guard_multiplier = 1.15 + 0.25 * min(temperature, 1.2)
    guard_tokens = int(round(estimated_tokens * guard_multiplier))
    guard_tokens += max(10, punctuation_count * 2)
    guard_tokens = max(min_tokens + 4, guard_tokens)

    # Allow a small headroom beyond the guard for natural sentence tails.
    tail_allowance = max(24, int(0.1 * estimated_tokens) + punctuation_count * 3)
    model_cap = guard_tokens + tail_allowance
    model_cap = min(model_cap, max_tokens_limit, max_model_len)

    # Confidence threshold for the log-prob stop detector: higher temperatures
    # reduce the bar slightly to reflect flatter distributions.
    stop_confidence = 0.55 + 0.25 * (1.0 - min(1.0, temperature))

    debug = {
        "clean_len": char_len,
        "word_like": word_like,
        "cjk_ratio": round(cjk_ratio, 4),
        "avg_chars_per_word": round(avg_chars_per_word, 3),
        "base_tpc": round(base_tpc, 3),
        "estimated_tokens": estimated_tokens,
        "language_hint": language_hint or None,
    }

    return PromptWindow(
        clean_text=clean,
        char_len=char_len,
        word_like=word_like,
        cjk_like=cjk_like,
        digit_count=digit_count,
        punctuation_count=punctuation_count,
        estimated_tokens=estimated_tokens,
        min_tokens=min_tokens,
        guard_tokens=guard_tokens,
        model_cap=model_cap,
        stop_confidence=stop_confidence,
        debug=debug,
    )


# --- Stop decision ----------------------------------------------------------


def _extract_logprob_entries(step: Mapping[Any, Any]) -> Iterable[Any]:
    """Yield log-prob objects from a sampling step mapping."""

    if isinstance(step, Mapping):
        return step.values()
    return []


def _detect_long_run(tokens: Sequence[int], *, min_run: int = 12) -> Optional[int]:
    run_length = 1
    for idx in range(1, len(tokens)):
        if tokens[idx] == tokens[idx - 1]:
            run_length += 1
            if run_length >= min_run:
                return idx - run_length + 1
        else:
            run_length = 1
    return None


def _detect_repeated_tail(tokens: Sequence[int], *, min_total: int = 24) -> Optional[int]:
    n = len(tokens)
    if n < min_total * 2:
        return None

    max_pattern = min(64, n // 2)
    for pattern in range(4, max_pattern + 1):
        tail = tokens[n - pattern : n]
        repeats = 0
        cursor = n - pattern
        while cursor - pattern >= 0 and tokens[cursor - pattern : cursor] == tail:
            repeats += 1
            cursor -= pattern
        if repeats >= 2 and pattern * (repeats + 1) >= min_total:
            return cursor + pattern
    return None


def apply_dynamic_stop(
    token_ids: Sequence[int],
    *,
    logprob_steps: Optional[Sequence[Optional[Mapping[Any, Any]]]] = None,
    window: PromptWindow,
    stop_token_id: int,
    tail_crop_tokens: int = 0,
    debug: bool = False,
) -> tuple[list[int], dict[str, Any]]:
    """Trim the generated token sequence according to dynamic EOS heuristics."""

    raw_len = len(token_ids)
    decision: dict[str, Any] = {
        "raw_len": raw_len,
        "estimated_tokens": window.estimated_tokens,
        "min_tokens": window.min_tokens,
        "guard_tokens": window.guard_tokens,
    }

    if raw_len == 0:
        decision.update({"cut_idx": 0, "reason": "empty"})
        return [], decision

    cut_idx = raw_len
    reason = "length"
    confidence: Optional[float] = None

    if logprob_steps:
        upto = min(len(logprob_steps), raw_len)
        for idx in range(upto):
            if idx < window.min_tokens:
                continue
            step = logprob_steps[idx]
            if not step:
                continue
            max_lp = -float("inf")
            stop_lp = None
            for entry in _extract_logprob_entries(step):
                lp = getattr(entry, "logprob", None)
                tok_id = getattr(entry, "token_id", None)
                if lp is None:
                    continue
                if lp > max_lp:
                    max_lp = lp
                if tok_id == stop_token_id:
                    stop_lp = lp
            if stop_lp is None or max_lp == -float("inf"):
                continue
            gap = max_lp - stop_lp
            conf = math.exp(-gap)
            if conf >= window.stop_confidence:
                cut_idx = min(cut_idx, idx)
                reason = "logprob"
                confidence = conf
                break

    if reason != "logprob":
        for idx, tok in enumerate(token_ids):
            if tok != stop_token_id:
                continue
            if idx < window.min_tokens:
                continue
            cut_idx = min(cut_idx, idx)
            reason = "stop_token"
            break

    # Detect degenerate repetitions or loops near the tail.
    run_start = _detect_long_run(token_ids, min_run=max(10, window.min_tokens // 2))
    if run_start is not None and run_start >= window.min_tokens:
        cut_idx = min(cut_idx, run_start)
        reason = "repeat_run"

    repeat_tail = _detect_repeated_tail(token_ids, min_total=max(24, window.min_tokens))
    if repeat_tail is not None and repeat_tail >= window.min_tokens:
        cut_idx = min(cut_idx, repeat_tail)
        reason = "repeat_tail"

    if raw_len > window.guard_tokens and window.guard_tokens < cut_idx:
        cut_idx = window.guard_tokens
        reason = "guard"

    cut_idx = max(0, min(cut_idx, raw_len))

    if tail_crop_tokens and cut_idx > window.min_tokens:
        if cut_idx - tail_crop_tokens >= window.min_tokens:
            cut_idx -= tail_crop_tokens
            reason = f"{reason}+tail"

    trimmed = [tok for tok in token_ids[:cut_idx] if tok != stop_token_id]

    # Avoid returning fewer than min_tokens if the raw sequence satisfied it.
    if raw_len >= window.min_tokens and len(trimmed) < window.min_tokens:
        trimmed = [tok for tok in token_ids[:window.min_tokens] if tok != stop_token_id]
        reason = f"{reason}|min_restore"
        cut_idx = window.min_tokens

    decision.update(
        {
            "cut_idx": cut_idx,
            "final_len": len(trimmed),
            "reason": reason,
            "confidence": confidence,
        }
    )

    if debug:
        print(f"[EOS] decision={decision} stats={window.debug}")

    return trimmed, decision

