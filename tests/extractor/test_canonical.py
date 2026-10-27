"""Person A: subjects(), normalise_facts(), default_importance()."""

from __future__ import annotations

import pytest

from sanitise.agents.extractor.canonical import (
    normalise_facts,
    normalise_relation,
    parse_relation,
    subjects,
)
from sanitise.agents.extractor.importance import IMPORTANCE, default_importance
from sanitise.schemas import RELATIONS, FactTuple, Importance, PIICategory, SensitiveSpan, Subject

C = PIICategory


def spans(text: str, *hits: tuple[str, PIICategory]) -> list[SensitiveSpan]:
    out = []
    for needle, cat in hits:
        start = text.index(needle)
        out.append(
            SensitiveSpan(
                text=needle, category=cat, start=start, end=start + len(needle), method="rule"
            )
        )
    return out


def by_id(subs: list[Subject]) -> dict[str, Subject]:
    return {s.id: s for s in subs}


# ---------------------------------------------------------------- subjects


def test_writer_always_first() -> None:
    subs = subjects("Is it okay to take paracetamol again?", [])
    assert [s.id for s in subs] == ["writer"]
    assert subs[0].label == "the writer"


def test_relative_of_writer_with_pronouns() -> None:
    text = "My son has stomach pain. He is 32 years old and his doctor is worried."
    subs = by_id(subjects(text, spans(text, ("My son", C.RELATION))))
    assert set(subs) == {"writer", "writer.son"}
    assert subs["writer.son"].label == "the writer's son"
    assert subs["writer.son"].mentions == ["My son", "He", "his"]


def test_writer_name_and_first_person() -> None:
    text = "Hi doctor, my name is Priya Raman. I am a surgeon and my ear rings."
    subs = by_id(subjects(text, spans(text, ("Priya Raman", C.NAME))))
    assert set(subs) == {"writer"}
    assert subs["writer"].mentions[:1] == ["Priya Raman"]
    assert {"I", "my"} <= set(subs["writer"].mentions)


def test_name_next_to_relation_joins_it() -> None:
    text = "My father, Cyril D'Souza, is 71 years old. He had a stroke."
    subs = by_id(subjects(text, spans(text, ("My father", C.RELATION), ("Cyril D'Souza", C.NAME))))
    assert set(subs) == {"writer", "writer.father"}
    assert subs["writer.father"].mentions == ["My father", "Cyril D'Souza", "He"]


def test_name_before_relation_joins_it() -> None:
    text = "Linda, my wife, has asthma."
    subs = by_id(subjects(text, spans(text, ("Linda", C.NAME), ("my wife", C.RELATION))))
    assert set(subs) == {"writer", "writer.wife"}
    assert subs["writer.wife"].mentions == ["Linda", "my wife"]


def test_email_sender_recipient_and_first_name() -> None:
    text = (
        "From: Daniel Whitcomb\nTo: Laura Fenn\n\nLaura, please call me before the meeting.\n\n"
        "Daniel"
    )
    hits = spans(
        text,
        ("Daniel Whitcomb", C.NAME),
        ("Laura Fenn", C.NAME),
    )
    laura = text.index("Laura,")
    daniel = text.rindex("Daniel")
    hits += [
        SensitiveSpan(text="Laura", category=C.NAME, start=laura, end=laura + 5, method="rule"),
        SensitiveSpan(text="Daniel", category=C.NAME, start=daniel, end=daniel + 6, method="rule"),
    ]
    subs = by_id(subjects(text, hits))
    assert set(subs) == {"writer", "person_1"}
    assert subs["writer"].mentions[:2] == ["Daniel Whitcomb", "Daniel"]
    assert subs["person_1"].label == "Laura Fenn"
    assert subs["person_1"].mentions == ["Laura Fenn", "Laura"]


def test_sign_off_name_is_writer() -> None:
    text = (
        "Hi Marta,\n\nPlease update the file for Kevin Oyelaran. He moved.\n\nThanks,\nHelen Park"
    )
    hits = spans(text, ("Marta", C.NAME), ("Kevin Oyelaran", C.NAME), ("Helen Park", C.NAME))
    subs = by_id(subjects(text, hits))
    assert subs["writer"].mentions[:1] == ["Helen Park"]
    assert subs["person_1"].mentions == ["Marta"]
    assert subs["person_2"].mentions == ["Kevin Oyelaran", "He"]


def test_relative_of_someone_else() -> None:
    text = "Samuel Ortega will be on leave. He is 51 years old. His wife will be the contact."
    hits = spans(text, ("Samuel Ortega", C.NAME), ("His wife", C.RELATION))
    subs = by_id(subjects(text, hits))
    assert "writer.wife" not in subs
    assert subs["person_2"].label == "Samuel Ortega's wife"
    assert subs["person_2"].mentions == ["His wife"]
    assert subs["person_1"].mentions == ["Samuel Ortega", "He"]


def test_pronoun_gender_follows_kinship() -> None:
    text = "My son and my daughter are ill. She has a fever and he has a cough."
    hits = spans(text, ("My son", C.RELATION), ("my daughter", C.RELATION))
    subs = by_id(subjects(text, hits))
    assert subs["writer.daughter"].mentions == ["my daughter", "She"]
    assert subs["writer.son"].mentions == ["My son", "he"]


def test_same_relative_twice_is_one_subject() -> None:
    text = "My husband is 63. Is my husband at risk?"
    first = text.index("My husband")
    second = text.index("my husband")
    hits = [
        SensitiveSpan(
            text=text[i : i + 10], category=C.RELATION, start=i, end=i + 10, method="rule"
        )
        for i in (first, second)
    ]
    subs = subjects(text, hits)
    assert [s.id for s in subs] == ["writer", "writer.husband"]
    assert subs[1].mentions == ["My husband", "my husband"]


def test_mentions_are_exact_strings_from_the_text() -> None:
    text = "My father, Cyril D'Souza, is 71. He had a stroke and his speech is slurred."
    hits = spans(text, ("My father", C.RELATION), ("Cyril D'Souza", C.NAME))
    for s in subjects(text, hits):
        for mention in s.mentions:
            assert mention in text


@pytest.mark.parametrize(
    "span, kin, possessive",
    [
        ("My son", "son", "my"),
        ("my 5 year old son", "son", "my"),
        ("my mother-in-law", "mother_in_law", "my"),
        ("Mom", "mother", None),
        ("His wife", "wife", "his"),
        ("my baby", "baby", "my"),
        ("our little girl", "daughter", "our"),
    ],
)
def test_parse_relation(span: str, kin: str, possessive: str | None) -> None:
    assert parse_relation(span) == (kin, possessive)


# ---------------------------------------------------------------- normalise_facts

SUBJECTS = [
    Subject(id="writer", label="the writer", mentions=["Priya Raman", "I"]),
    Subject(id="writer.son", label="the writer's son", mentions=["My son", "He"]),
    Subject(id="person_1", label="Laura Fenn", mentions=["Laura Fenn", "Laura"]),
]


def fact(subject: str, relation: str, value: str, importance: str = "medium") -> FactTuple:
    return FactTuple(subject=subject, relation=relation, value=value, importance=importance)


@pytest.mark.parametrize(
    "given, expected",
    [
        ("lives in", "lives_in"),
        ("resides_in", "lives_in"),
        ("Diagnosed With", "has_condition"),
        ("occupation", "works_as"),
        ("medication", "took_medication"),
        ("symptom duration", "symptom_duration"),
        ("duration_of_pain", "symptom_duration"),
        ("phone number", "contact"),
        ("has_symptom", "has_symptom"),
        ("favourite colour", "other"),
    ],
)
def test_normalise_relation(given: str, expected: str) -> None:
    assert normalise_relation(given) == expected


def test_every_relation_maps_to_itself() -> None:
    assert all(normalise_relation(r) == r for r in RELATIONS)


def test_subjects_resolved_from_labels_and_mentions() -> None:
    facts = [
        fact("writer.son", "age", "32 years old"),
        fact("the writer's son", "has_symptom", "abdominal pain", "high"),
        fact("My son", "weight", "220 pounds"),
        fact("patient", "works_as", "surgeon"),
        fact("Laura", "contact", "555-0142"),
        fact("writer son", "height", "6 ft 3 inches"),
    ]
    out = normalise_facts(facts, SUBJECTS)
    assert [f.subject for f in out] == [
        "writer.son",
        "writer.son",
        "writer.son",
        "writer",
        "person_1",
        "writer.son",
    ]


def test_unknown_subject_dropped() -> None:
    out = normalise_facts([fact("the doctor", "other", "said rest")], SUBJECTS)
    assert out == []


def test_values_trimmed_and_empty_dropped() -> None:
    out = normalise_facts(
        [fact("writer", "has_symptom", '  "dry   cough."  '), fact("writer", "other", " ... ")],
        SUBJECTS,
    )
    assert [f.value for f in out] == ["dry cough"]


def test_duplicates_merged_keeping_highest_importance() -> None:
    out = normalise_facts(
        [
            fact("writer", "has_condition", "Vertigo", "medium"),
            fact("I", "diagnosis", "vertigo.", "high"),
            fact("writer", "has_condition", "the vertigo", "low"),
            fact("writer", "has_symptom", "vertigo", "low"),  # different relation: kept
        ],
        SUBJECTS,
    )
    assert [(f.relation, f.value, f.importance) for f in out] == [
        ("has_condition", "Vertigo", Importance.HIGH),
        ("has_symptom", "vertigo", Importance.LOW),
    ]


def test_normalise_is_idempotent() -> None:
    facts = [fact("My son", "lives in", " Port Isaac "), fact("writer", "job", "surgeon")]
    once = normalise_facts(facts, SUBJECTS)
    assert normalise_facts(once, SUBJECTS) == once


# ---------------------------------------------------------------- importance


def test_importance_table_covers_vocabulary() -> None:
    assert set(IMPORTANCE) == set(RELATIONS)


@pytest.mark.parametrize(
    "relation, expected",
    [
        ("has_condition", Importance.HIGH),
        ("has_symptom", Importance.HIGH),
        ("took_medication", Importance.HIGH),
        ("lives_in", Importance.MEDIUM),
        ("age", Importance.MEDIUM),
        ("contact", Importance.LOW),
        ("identifier", Importance.LOW),
        ("diagnosed_with", Importance.HIGH),  # synonym
        ("mood", Importance.LOW),  # unknown -> other
    ],
)
def test_default_importance(relation: str, expected: Importance) -> None:
    assert default_importance(relation) is expected
