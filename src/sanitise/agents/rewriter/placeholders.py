"""Agent 3 - placeholder allocation.  OWNER: Person A (Deterministic Logic).

PLACEHOLDER created by Person C (already close to final). Keep the signature.

``allocate_placeholders`` gives every decision with ``action == placeholder`` a token such
as ``[PERSON_1]`` and writes it into ``decision.replacement``. The same original text gets the
same token within one document; tokens restart at 1 in every document, so they can never be
used to join two documents. Returns ``{token: original text}``.
"""

from __future__ import annotations

from sanitise.schemas import Action, PIICategory, RoutingDecision

C = PIICategory

PREFIX: dict[PIICategory, str] = {
    C.NAME: "PERSON",
    C.PHONE: "PHONE",
    C.EMAIL: "EMAIL",
    C.ID_NUMBER: "ID",
    C.ADDRESS: "ADDRESS",
}


def allocate_placeholders(decisions: list[RoutingDecision]) -> dict[str, str]:
    tokens: dict[str, str] = {}  # token -> original
    by_text: dict[tuple[str, str], str] = {}  # (prefix, normalised original) -> token
    counters: dict[str, int] = {}
    for decision in decisions:
        if decision.action is not Action.PLACEHOLDER:
            continue
        prefix = PREFIX.get(decision.category, "REDACTED")
        key = (prefix, " ".join(decision.span_text.lower().split()))
        if key not in by_text:
            counters[prefix] = counters.get(prefix, 0) + 1
            token = f"[{prefix}_{counters[prefix]}]"
            by_text[key] = token
            tokens[token] = decision.span_text
        decision.replacement = by_text[key]
    return tokens
