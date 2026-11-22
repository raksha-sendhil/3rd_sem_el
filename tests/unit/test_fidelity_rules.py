"""Keyed identifiers, who said what, and the rewrite's fidelity checks."""

from __future__ import annotations

import pytest

from sanitise.agents.extractor import speakers
from sanitise.agents.rewriter import fidelity
from sanitise.rules.keyed import keyed_identifiers
from sanitise.schemas import FactTuple, Importance

TEXT = (
    "Patient: I have had sharp chest pain when climbing stairs for a week.\n"
    "Doctor: This may be angina, so get a treadmill test before climbing again."
)


@pytest.mark.parametrize(
    "text, value",
    [
        ("The SWIFT BIC for the provider is QPJYIDKL73.", "QPJYIDKL73"),
        ("swift bic for the payment was KLFRUA3J90X", "KLFRUA3J90X"),
        ("Policy number: AB-993812", "AB-993812"),
    ],
)
def test_labelled_identifiers_are_found(text: str, value: str) -> None:
    assert [s.text for s in keyed_identifiers(text, [])] == [value]


def test_each_fact_gets_its_speaker() -> None:
    facts = [
        FactTuple(subject="writer", relation=r, value=v, importance=Importance.HIGH)
        for r, v in [("has_symptom", "sharp chest pain"), ("plan", "get a treadmill test")]
    ]
    assert [f.said_by for f in speakers.attribute(TEXT, facts)] == ["patient", "doctor"]


def test_first_person_is_rendered_in_the_third_person() -> None:
    fact = FactTuple(
        subject="writer", relation="plan", value="get a treadmill test",
        importance=Importance.HIGH, said_by="doctor",
    )  # fmt: skip
    safe = [(fact, fact.value)]
    assert fidelity.problems("I will get a treadmill test.", safe)
    fixed = fidelity.repair("I will get a treadmill test.", safe, {"writer": "the patient"})
    assert fixed == "The doctor advised: get a treadmill test."
