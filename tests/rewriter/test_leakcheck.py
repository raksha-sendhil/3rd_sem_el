"""Person A: leak check variants and the deterministic mask fallback."""

from __future__ import annotations

import pytest

from sanitise.agents.rewriter.leakcheck import leak_check, mask_fallback
from sanitise.schemas import Action, PIICategory, RoutingDecision

C = PIICategory


def removed(text: str, cat: PIICategory, action: Action = Action.PLACEHOLDER) -> RoutingDecision:
    return RoutingDecision(span_text=text, category=cat, action=action, reason="r", start=0,
                           end=len(text))  # fmt: skip


@pytest.mark.parametrize(
    "original, category, safe, leaked",
    [
        ("Priya Raman", C.NAME, "Ask priya  raman.", "priya  raman"),
        ("Laura Fenn", C.NAME, "Laura, please call.", "Laura"),
        ("Laura Fenn", C.NAME, "Ms Fenn called.", "Fenn"),
        ("+91 98450 12345", C.PHONE, "Ring 98450-12345 now.", "98450-12345"),
        ("(415) 555-0142", C.PHONE, "or 415.555.0142", "415.555.0142"),
        ("HC-2291746", C.ID_NUMBER, "patient HC 2291746", "HC 2291746"),
        ("HC-2291746", C.ID_NUMBER, "number 2291746 on file", "2291746"),
        ("rajesh.kumar84@examplemail.com", C.EMAIL, "user rajesh.kumar84 wrote", "rajesh.kumar84"),
        ("1400 Smith Street, Houston, TX 77002", C.ADDRESS, "at 1400 Smith St", "1400 Smith"),
        (
            "1400 Smith Street, Houston, TX 77002",
            C.ADDRESS,
            "at 1400 smith street",
            "1400 smith street",
        ),
        ("Fortis Hospital, Mulund", C.LOCATION, "admitted in Mulund", "Mulund"),
        ("32 years old", C.AGE, "a 32-year-old man", "32-year"),
        ("28", C.AGE, "aged 28, she", "aged 28"),
        ("3 March 2022", C.DATE, "on March 3, 2022", "March 3"),
        ("3 March 2022", C.DATE, "on 03/03/2022", "03/03/2022"),
        ("approximately 220 pounds", C.PHYSICAL, "weighs 220 lbs", "220 lbs"),
        ("6 ft 3 inches", C.PHYSICAL, "he is 6ft 3 tall", "6ft 3"),
        ("tea taster", C.OCCUPATION, "works as a Tea-Taster", "Tea-Taster"),
    ],
)
def test_variants_found(
    original: str, category: PIICategory, safe: str, leaked: str | None
) -> None:
    leaks = leak_check([removed(original, category)], safe)
    if leaked is None:
        assert leaks == []
        return
    assert [leak.text for leak in leaks] == [leaked]
    leak = leaks[0]
    assert safe[leak.start : leak.end] == leak.text
    assert leak.category is category and leak.method == "rule"


@pytest.mark.parametrize(
    "original, category, safe",
    [
        ("28", C.AGE, "for 28 days"),  # a bare number needs an age word
        ("Laura Fenn", C.NAME, "laura is a word here"),  # a lone part must be capitalised
        ("Al Smith", C.NAME, "Al said"),  # parts shorter than three letters are ignored
        ("2015", C.DATE, "in the 2010s"),
        ("Priya Raman", C.NAME, "Call [PERSON_1] today"),
    ],
)
def test_no_false_leaks(original: str, category: PIICategory, safe: str) -> None:
    assert leak_check([removed(original, category)], safe) == []


def test_kept_decisions_are_ignored() -> None:
    keep = removed("my son", C.RELATION, Action.KEEP)
    assert leak_check([keep], "my son is ill") == []


def test_overlapping_matches_reported_once() -> None:
    decisions = [removed("Laura Fenn", C.NAME), removed("Fenn", C.NAME)]
    leaks = leak_check(decisions, "Laura Fenn called")
    assert [leak.text for leak in leaks] == ["Laura Fenn"]


def test_mask_fallback_uses_tokens_and_passes_the_check() -> None:
    decisions = [
        removed("Daniel Whitcomb", C.NAME),
        removed("+91 98450 12345", C.PHONE),
        removed("Coonoor", C.LOCATION, Action.GENERALISE),
    ]
    tokens = {"[PERSON_1]": "Daniel Whitcomb", "[PHONE_1]": "+91 98450 12345"}
    safe = "Daniel lives in Coonoor; call 98450 12345 or daniel whitcomb."
    leaks = leak_check(decisions, safe)
    masked = mask_fallback(safe, leaks, tokens)
    assert masked == "[PERSON_1] lives in [REDACTED]; call [PHONE_1] or [PERSON_1]."
    assert leak_check(decisions, masked) == []


def test_mask_fallback_no_leaks_is_identity() -> None:
    assert mask_fallback("clean text", [], {}) == "clean text"
