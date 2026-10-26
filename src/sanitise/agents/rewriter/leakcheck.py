"""Agent 3 - leak check and fallback masking.  OWNER: Person A (Deterministic Logic).

PLACEHOLDER created by Person C: exact, case-insensitive matching only. Replace the bodies;
keep the signatures.

* ``leak_check(removed, safe_text)``: ``removed`` are the decisions whose action is
  placeholder or generalise. Return one ``SensitiveSpan`` (offsets in ``safe_text``,
  ``method="rule"``) for every place their original text still appears. The real version
  also catches close variants: different spacing or punctuation, a first name on its own
  when the full name was removed, digits of a phone number with different separators.
* ``mask_fallback(safe_text, leaks, placeholders)``: replace every leaked span
  deterministically (with its placeholder token if it has one, otherwise ``[REDACTED]``).
  The pipeline uses it when a rewrite still leaks after one retry.
"""

from __future__ import annotations

import re

from sanitise.schemas import Action, RoutingDecision, SensitiveSpan


def leak_check(removed: list[RoutingDecision], safe_text: str) -> list[SensitiveSpan]:
    leaks: list[SensitiveSpan] = []
    for decision in removed:
        if decision.action is Action.KEEP or not decision.span_text.strip():
            continue
        pattern = r"(?<!\w)" + re.escape(decision.span_text.strip()) + r"(?!\w)"
        for m in re.finditer(pattern, safe_text, re.IGNORECASE):
            leaks.append(
                SensitiveSpan(
                    text=safe_text[m.start() : m.end()],
                    category=decision.category,
                    start=m.start(),
                    end=m.end(),
                    method="rule",
                    rationale=f"leak check: removed {decision.category.value} still present",
                )
            )
    return leaks


def mask_fallback(safe_text: str, leaks: list[SensitiveSpan], placeholders: dict[str, str]) -> str:
    token_for = {original.lower(): token for token, original in placeholders.items()}
    out = safe_text
    for leak in sorted(leaks, key=lambda s: s.start or 0, reverse=True):
        assert leak.start is not None and leak.end is not None
        out = out[: leak.start] + token_for.get(leak.text.lower(), "[REDACTED]") + out[leak.end :]
    return out
