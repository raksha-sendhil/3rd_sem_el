"""Stage 6: combine the rule pass with the LLM quasi-identifier pass."""

import re

from sanitise.rules.normalise import is_positional, resolve_overlaps, trim, union_same_category
from sanitise.schemas import SensitiveSpan

# How an LLM combination lists its parts: "left-sided abdominal + bilateral pelvic pain"
_PART_SPLIT = re.compile(r"\s*(?:\+|&|;|,|\band\b|\bwith\b|\bplus\b)\s*", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def _norm(s: str) -> str:
    return " " + _NON_ALNUM.sub(" ", s.lower()).strip() + " "


def restates_explicit(span: SensitiveSpan, explicit_norms: list[str]) -> bool:
    """True when every part of the LLM span is already covered by a rule span's text,
    i.e. the 'quasi-identifier' adds nothing beyond identifiers we already remove."""
    parts = [_norm(p) for p in _PART_SPLIT.split(span.text)]
    parts = [p for p in parts if p.strip()]
    if not parts:
        return True
    return all(any(p in e for e in explicit_norms) for p in parts)


def merge_spans(
    rule_spans: list[SensitiveSpan], llm_spans: list[SensitiveSpan]
) -> list[SensitiveSpan]:
    """Union, drop quasi-identifiers that restate an explicit identifier,
    resolve overlaps by category precedence.

    Returns positional spans sorted by offset, followed by the non-positional LLM
    spans in the order the model produced them (deduplicated by category + text).
    """
    explicit_norms = [_norm(s.text) for s in rule_spans]

    positional = [s for s in rule_spans if is_positional(s)]
    inferred: list[SensitiveSpan] = []
    seen: set[tuple[str, str]] = set()
    for s in llm_spans:
        if not s.text.strip() or restates_explicit(s, explicit_norms):
            continue
        if is_positional(s):
            positional.append(s)
            continue
        key = (s.category.value, _norm(s.text))
        if key in seen:
            continue
        seen.add(key)
        inferred.append(s)

    positional = [t for s in positional if (t := trim(s)) is not None]
    if positional:
        positional = resolve_overlaps(union_same_category(positional))
    positional.sort(key=lambda s: (s.start, s.end))
    return positional + inferred
