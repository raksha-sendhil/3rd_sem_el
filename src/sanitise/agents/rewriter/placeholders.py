"""Agent 3 - placeholder allocation.  OWNER: Person A (Deterministic Logic).

``allocate_placeholders`` gives every decision with ``action == placeholder`` a token and
writes it into ``decision.replacement``. Returns ``{token: original text}``.

* Names take their prefix from the party role routing copied into the decision's reason
  (``role=client`` -> ``[CLIENT_1]``, ``party`` -> ``[PARTY_1]``, ``witness`` ->
  ``[WITNESS_1]``, no role -> ``[PERSON_1]``), as in the legal build spec.
* The same original text gets the same token within one document, and so does a lone first
  name or surname of a full name in the same document ("Mehta" -> the token of "Rohan Mehta").
  When a name has mentions with different roles, the first role found wins.
* Tokens restart at 1 in every document, so they can never be used to join two documents.
"""

from __future__ import annotations

from sanitise.rules.roles import name_words, role_of
from sanitise.schemas import Action, PIICategory, RoutingDecision

C = PIICategory

PREFIX: dict[PIICategory, str] = {
    C.NAME: "PERSON",
    C.PHONE: "PHONE",
    C.EMAIL: "EMAIL",
    C.ID_NUMBER: "ID",
    C.ADDRESS: "ADDRESS",
    C.FINANCIAL: "ACCOUNT",
    C.CASE_NUMBER: "CASE",
    C.MATTER_CODENAME: "MATTER",
    C.ORGANIZATION: "ORG",
    C.COURT: "COURT",
    C.MONEY: "AMOUNT",
}
ROLE_PREFIX = {"client": "CLIENT", "party": "PARTY", "witness": "WITNESS"}


def _name_key(text: str) -> tuple[str, ...]:
    return tuple(name_words(text)) or (" ".join(text.lower().split()),)


def _canonical_names(names: list[RoutingDecision]) -> dict[tuple[str, ...], tuple[str, ...]]:
    """Each name's key -> the key of the full name it belongs to (itself if none or ambiguous)."""
    keys = {_name_key(d.span_text) for d in names}
    full = [k for k in keys if len(k) > 1]
    out = {}
    for key in keys:
        owners = [f for f in full if len(key) == 1 and key[0] in f]
        out[key] = owners[0] if len(owners) == 1 else key
    return out


def allocate_placeholders(decisions: list[RoutingDecision]) -> dict[str, str]:
    placeholder = [d for d in decisions if d.action is Action.PLACEHOLDER]
    names = [d for d in placeholder if d.category is C.NAME]
    canonical = _canonical_names(names)

    role: dict[tuple[str, ...], str] = {}  # canonical name key -> role, first found wins
    original: dict[tuple[str, ...], str] = {}  # canonical name key -> first full spelling
    for d in names:
        key = canonical[_name_key(d.span_text)]
        if (r := role_of(d.reason)) and key not in role:
            role[key] = r
        words = len(_name_key(d.span_text))
        if key not in original or words > len(_name_key(original[key])):
            original[key] = d.span_text

    tokens: dict[str, str] = {}  # token -> original
    by_key: dict[tuple[str, ...], str] = {}  # (prefix, *normalised original) -> token
    counters: dict[str, int] = {}
    for decision in placeholder:
        if decision.category is C.NAME:
            name = canonical[_name_key(decision.span_text)]
            prefix = ROLE_PREFIX.get(role.get(name, ""), PREFIX[C.NAME])
            key, shown = (prefix, *name), original[name]
        else:
            prefix = PREFIX.get(decision.category, "REDACTED")
            key, shown = (prefix, " ".join(decision.span_text.lower().split())), decision.span_text
        if key not in by_key:
            counters[prefix] = counters.get(prefix, 0) + 1
            token = f"[{prefix}_{counters[prefix]}]"
            by_key[key] = token
            tokens[token] = shown
        decision.replacement = by_key[key]
    return tokens
