"""Person A: the routing policy (plan 6.6) and QUASI_COMBO handling."""

from __future__ import annotations

import pytest

from sanitise.agents.rewriter.routing import DEFAULT_ACTION, combo_members, route
from sanitise.schemas import Action, FactTuple, PIICategory, SensitiveSpan

C = PIICategory
A = Action


def positional(text: str, needle: str, cat: PIICategory) -> SensitiveSpan:
    start = text.index(needle)
    return SensitiveSpan(
        text=needle, category=cat, start=start, end=start + len(needle), method="rule"
    )


def combo(text: str) -> SensitiveSpan:
    return SensitiveSpan(text=text, category=C.QUASI_COMBO, method="llm", rationale="r")


@pytest.mark.parametrize(
    "cat, action",
    [
        (C.NAME, A.PLACEHOLDER),
        (C.PHONE, A.PLACEHOLDER),
        (C.EMAIL, A.PLACEHOLDER),
        (C.ID_NUMBER, A.PLACEHOLDER),
        (C.ADDRESS, A.PLACEHOLDER),
        (C.AGE, A.GENERALISE),
        (C.DATE, A.GENERALISE),
        (C.LOCATION, A.GENERALISE),
        (C.OCCUPATION, A.GENERALISE),
        (C.PHYSICAL, A.GENERALISE),
        (C.SYMPTOM, A.KEEP),
        (C.RELATION, A.KEEP),
    ],
)
def test_default_table(cat: PIICategory, action: Action) -> None:
    assert DEFAULT_ACTION[cat] is action


def test_one_decision_per_span_same_order_offsets_copied() -> None:
    text = "I am Priya Raman, 34 years old, from Coonoor. Call +91 98450 12345."
    spans = [
        positional(text, "Priya Raman", C.NAME),
        positional(text, "34 years old", C.AGE),
        positional(text, "Coonoor", C.LOCATION),
        positional(text, "+91 98450 12345", C.PHONE),
        combo("Coonoor + 34"),
    ]
    decisions = route(spans, [])
    assert len(decisions) == len(spans)
    for span, decision in zip(spans, decisions, strict=True):
        assert (decision.span_text, decision.category) == (span.text, span.category)
        assert (decision.start, decision.end) == (span.start, span.end)
        assert decision.reason
    assert [d.action for d in decisions] == [A.PLACEHOLDER, A.GENERALISE, A.GENERALISE,
                                             A.PLACEHOLDER, A.GENERALISE]  # fmt: skip


def test_generalised_replacements_filled_placeholders_left() -> None:
    text = "Priya Raman is 34 years old."
    decisions = route(
        [positional(text, "Priya Raman", C.NAME), positional(text, "34 years old", C.AGE)], []
    )
    assert decisions[0].replacement is None  # allocate_placeholders fills it
    assert decisions[1].replacement == "in their 30s"


def test_common_condition_kept_rare_generalised() -> None:
    text = "Diagnosed with type 2 diabetes and amyloidosis."
    decisions = route(
        [
            positional(text, "type 2 diabetes", C.MEDICAL_CONDITION),
            positional(text, "amyloidosis", C.MEDICAL_CONDITION),
        ],
        [],
    )
    assert decisions[0].action is A.KEEP
    assert decisions[1].action is A.GENERALISE
    assert "rare" in decisions[1].reason
    assert decisions[1].replacement == "a rare protein disorder"


def test_combo_members_by_text() -> None:
    text = "I am a lighthouse keeper in Port Isaac with a rash."
    spans = [
        positional(text, "lighthouse keeper", C.OCCUPATION),
        positional(text, "Port Isaac", C.LOCATION),
        positional(text, "rash", C.SYMPTOM),
        combo("lighthouse keeper + Port Isaac"),
    ]
    assert combo_members(spans[3], spans) == [0, 1]


def test_combo_members_by_described_category() -> None:
    text = "My daughter is 6 years old, attends Greenfield School in Tiruppur since 12 May 2023."
    spans = [
        positional(text, "6 years old", C.AGE),
        positional(text, "Greenfield School", C.LOCATION),
        positional(text, "Tiruppur", C.LOCATION),
        positional(text, "12 May 2023", C.DATE),
        combo("child age + named school + small town + precise onset date"),
    ]
    assert combo_members(spans[4], spans) == [0, 1, 2, 3]


def test_combo_members_climb_two_steps() -> None:
    text = "A 34 year old pediatric neurosurgeon in Coonoor with vertigo."
    spans = [
        positional(text, "34 year old", C.AGE),
        positional(text, "pediatric neurosurgeon", C.OCCUPATION),
        positional(text, "Coonoor", C.LOCATION),
        positional(text, "vertigo", C.MEDICAL_CONDITION),
        combo("rare occupation (pediatric neurosurgeon) + small town (Coonoor) + vertigo"),
    ]
    d = route(spans, [])
    assert d[0].replacement == "in their 30s"  # not a member: one step
    assert d[1].replacement == "a working professional"  # member: two steps
    assert "two steps" in d[1].reason
    assert d[3].action is A.GENERALISE  # condition in a combination
    assert d[3].replacement == "a balance or hearing problem"  # ... one step only


def test_all_keep_combo_is_broken_by_least_important_member() -> None:
    text = "My son has left-sided abdominal pain and bilateral pelvic pain."
    spans = [
        positional(text, "My son", C.RELATION),
        positional(text, "left-sided abdominal pain", C.SYMPTOM),
        positional(text, "bilateral pelvic pain", C.SYMPTOM),
        combo("left-sided abdominal pain + bilateral pelvic pain"),
    ]
    facts = [
        FactTuple(subject="writer.son", relation="has_symptom",
                  value="left-sided abdominal pain", importance="high"),
        FactTuple(subject="writer.son", relation="has_symptom",
                  value="bilateral pelvic pain", importance="medium"),
    ]  # fmt: skip
    d = route(spans, facts)
    assert [x.action for x in d[:3]] == [A.KEEP, A.KEEP, A.GENERALISE]
    assert d[2].replacement == "pelvic pain"
    assert "break the combination" in d[2].reason


def test_combo_without_any_member_changes_nothing() -> None:
    text = "I have a cough."
    spans = [positional(text, "cough", C.SYMPTOM), combo("rare job + rare hobby")]
    assert route(spans, [])[0].action is A.KEEP


def test_empty() -> None:
    assert route([], []) == []
