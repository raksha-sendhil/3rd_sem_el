"""Agent 3 - placeholder allocation.  OWNER: Person A (Deterministic Logic).

``allocate_placeholders`` gives every decision with ``action == placeholder`` a token such
as ``[PERSON_1]`` and writes it into ``decision.replacement``. Returns ``{token: original}``.

* The same detail gets the same token within one document, however it is written: names
  ignoring case, and a first or last name on its own joins the full name it is part of
  ("Laura" and "Laura Fenn" are both ``[PERSON_2]``, mapped to the full name); phone numbers
  by their digits ("(415) 555-0142" and "415.555.0142"); everything else by its letters and
  digits, ignoring case and punctuation.
* Tokens are numbered per category in order of first appearance and restart at 1 in every
  document, so a token can never be used to join two documents.
"""

from __future__ import annotations

import re

from sanitise.schemas import Action, PIICategory, RoutingDecision

C = PIICategory

PREFIX: dict[PIICategory, str] = {
    C.NAME: "PERSON",
    C.PHONE: "PHONE",
    C.EMAIL: "EMAIL",
    C.ID_NUMBER: "ID",
    C.ADDRESS: "ADDRESS",
}


def _identity(category: PIICategory, text: str) -> str:
    if category is C.PHONE:
        digits = re.sub(r"\D", "", text)
        return digits[-10:] if len(digits) >= 7 else text.lower()
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _name_parts(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z][a-z'\u2019-]*", text.lower()) if len(w) > 1)


def allocate_placeholders(decisions: list[RoutingDecision]) -> dict[str, str]:
    tokens: dict[str, str] = {}  # token -> original
    by_identity: dict[tuple[str, str], str] = {}  # (prefix, identity) -> token
    names: list[tuple[frozenset[str], str]] = []  # (name parts, token)
    counters: dict[str, int] = {}
    for decision in decisions:
        if decision.action is not Action.PLACEHOLDER:
            continue
        prefix = PREFIX.get(decision.category, "REDACTED")
        key = (prefix, _identity(decision.category, decision.span_text))
        token = by_identity.get(key)
        if token is None and decision.category is C.NAME:
            parts = _name_parts(decision.span_text)
            matches = {t for p, t in names if parts and (parts <= p or p <= parts)}
            if len(matches) == 1:  # ambiguous ("D'Souza" of two D'Souzas) gets its own token
                token = matches.pop()
                if len(decision.span_text) > len(tokens[token]):
                    tokens[token] = decision.span_text  # map to the fuller name
        if token is None:
            counters[prefix] = counters.get(prefix, 0) + 1
            token = f"[{prefix}_{counters[prefix]}]"
            tokens[token] = decision.span_text
        by_identity[key] = token
        if decision.category is C.NAME:
            names.append((_name_parts(decision.span_text), token))
        decision.replacement = token
    return tokens
