"""Person A: placeholder allocation."""

from __future__ import annotations

from sanitise.agents.rewriter.placeholders import allocate_placeholders
from sanitise.schemas import Action, PIICategory, RoutingDecision

C = PIICategory


def ph(text: str, cat: PIICategory = C.NAME) -> RoutingDecision:
    return RoutingDecision(span_text=text, category=cat, action=Action.PLACEHOLDER, reason="r")


def test_tokens_per_category_in_order() -> None:
    decisions = [ph("Priya Raman"), ph("+91 98450 12345", C.PHONE), ph("Arjun Nair"),
                 ph("a@b.com", C.EMAIL), ph("HC-1", C.ID_NUMBER), ph("1 Elm St", C.ADDRESS),
                 ph("www.example.org", C.OTHER)]  # fmt: skip
    tokens = allocate_placeholders(decisions)
    assert [d.replacement for d in decisions] == [
        "[PERSON_1]", "[PHONE_1]", "[PERSON_2]", "[EMAIL_1]", "[ID_1]", "[ADDRESS_1]",
        "[REDACTED_1]",
    ]  # fmt: skip
    assert tokens["[PERSON_2]"] == "Arjun Nair"


def test_same_detail_same_token() -> None:
    decisions = [ph("Laura Fenn"), ph("laura  fenn"), ph("(415) 555-0142", C.PHONE),
                 ph("415.555.0142", C.PHONE)]  # fmt: skip
    tokens = allocate_placeholders(decisions)
    assert [d.replacement for d in decisions] == [
        "[PERSON_1]", "[PERSON_1]", "[PHONE_1]", "[PHONE_1]",
    ]  # fmt: skip
    assert tokens == {"[PERSON_1]": "Laura Fenn", "[PHONE_1]": "(415) 555-0142"}


def test_first_name_joins_full_name_either_order() -> None:
    decisions = [ph("Daniel"), ph("Daniel Whitcomb"), ph("Laura Fenn"), ph("Laura")]
    tokens = allocate_placeholders(decisions)
    assert [d.replacement for d in decisions] == [
        "[PERSON_1]", "[PERSON_1]", "[PERSON_2]", "[PERSON_2]",
    ]  # fmt: skip
    assert tokens["[PERSON_1]"] == "Daniel Whitcomb"  # mapped to the fuller name


def test_ambiguous_surname_gets_its_own_token() -> None:
    decisions = [ph("Anita D'Souza"), ph("Cyril D'Souza"), ph("D'Souza")]
    allocate_placeholders(decisions)
    assert [d.replacement for d in decisions] == ["[PERSON_1]", "[PERSON_2]", "[PERSON_3]"]


def test_non_placeholder_decisions_untouched() -> None:
    keep = RoutingDecision(span_text="my son", category=C.RELATION, action=Action.KEEP, reason="r")
    assert allocate_placeholders([keep]) == {}
    assert keep.replacement is None


def test_tokens_restart_in_every_document() -> None:
    first = allocate_placeholders([ph("Priya Raman")])
    second = allocate_placeholders([ph("Arjun Nair")])
    assert list(first) == list(second) == ["[PERSON_1]"]
