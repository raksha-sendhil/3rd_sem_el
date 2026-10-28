"""Agent 2's identity facts and record check, and Agent 3's grounding check: no model needed."""

from __future__ import annotations

import pytest

from sanitise.agents.extractor.canonical import subjects
from sanitise.agents.extractor.pipeline import identity_facts, in_record
from sanitise.agents.pipeline import sanitise
from sanitise.agents.rewriter.grounding import REDACTED, mask, ungrounded
from sanitise.llm import MockLLM
from sanitise.schemas import PIICategory, SensitiveSpan
from tests.conftest import SAMPLE_TEXT

C = PIICategory


def _span(text: str, needle: str, cat: PIICategory) -> SensitiveSpan:
    start = text.index(needle)
    return SensitiveSpan(
        text=needle, category=cat, start=start, end=start + len(needle), method="rule"
    )


def _identity(text: str, hits: list[tuple[str, PIICategory]]) -> set[tuple[str, str, str]]:
    spans = [_span(text, needle, cat) for needle, cat in hits]
    return {
        (f.subject, f.relation, f.value) for f in identity_facts(text, subjects(text, spans), spans)
    }


def test_an_id_number_belongs_to_the_person_mentioned_before_it() -> None:
    text = (
        "My name is Mira Kale. My husband has had a cough for a month. "
        "His patient ID is HC-5512093."
    )
    got = _identity(
        text,
        [("My husband", C.RELATION), ("HC-5512093", C.ID_NUMBER), ("Mira Kale", C.NAME)],
    )
    assert ("writer.husband", "identifier", "HC-5512093") in got
    assert ("writer.husband", "relation_to", "husband") in got
    assert ("writer", "identifier", "Mira Kale") in got


def test_contact_details_default_to_the_writer() -> None:
    text = "My daughter has a rash. You can reach me on 98450 11223 or at m.k@post.test."
    got = _identity(
        text,
        [("My daughter", C.RELATION), ("98450 11223", C.PHONE), ("m.k@post.test", C.EMAIL)],
    )
    assert ("writer", "contact", "98450 11223") in got
    assert ("writer", "contact", "m.k@post.test") in got


def test_identity_facts_never_contain_words_that_are_not_detected_details() -> None:
    text = "I am the writer and my husband is unwell. Let me know."
    assert _identity(text, [("my husband", C.RELATION)]) == {
        ("writer.husband", "relation_to", "husband")
    }


@pytest.mark.parametrize(
    "value, kept",
    [
        ("chest pain when walking", True),
        ("two weeks", True),
        ("student", False),  # inferred, never written
        ("HC-229174174", False),  # an ID copied wrongly
        ("44", True),
        ("in my 40s", False),  # not the record's wording
    ],
)
def test_model_facts_must_be_worded_from_the_record(value: str, kept: bool) -> None:
    text = "I am 44. I get chest pain when walking, for two weeks now. My ID is HC-2291746."
    assert in_record(value, text) is kept


FACTS_TEXT = (
    "the writer - identifier: [PERSON_1].\n"
    "the writer - age: in their 40s.\n"
    "the writer - lives in: a town in the region.\n"
    "the writer - has condition: Crohn's disease.\n"
    "the writer - contact: [EMAIL_1]."
)


@pytest.mark.parametrize(
    "rewrite, flagged",
    [
        ("I am [PERSON_1], in my 40s, and I have Crohn's disease. Email [EMAIL_1].", []),
        ("Hello Doctor, I am [PERSON_1] and I live in a town in the region.", []),
        ("I am Helen Park, in my 40s.", ["Helen Park"]),
        ("Helen Park here: I am in my 40s.", ["Helen Park"]),
        ("I live in Pune and have Crohn's disease.", ["Pune"]),
        ("My patient ID is HC-229174174.", ["ID", "HC", "229174174"]),
        ("I was born in 1981.", ["1981"]),
        ("Write to me at h.park@mail.test.", ["h.park@mail.test"]),
    ],
)
def test_grounding_flags_specific_words_that_are_not_in_the_facts(
    rewrite: str, flagged: list[str]
) -> None:
    assert [s.text for s in ungrounded(rewrite, FACTS_TEXT)] == flagged


def test_mask_replaces_every_flagged_span() -> None:
    text = "I live in Pune, born 1981."
    masked = mask(text, ungrounded(text, FACTS_TEXT))
    assert masked == f"I live in {REDACTED}, born {REDACTED}."
    assert ungrounded(masked, FACTS_TEXT) == []


async def test_a_made_up_name_is_retried_then_masked(fake_rules: None) -> None:
    invented = "My son, Arthur Penrose, is in his 30s and works as a lighthouse keeper."
    llm = MockLLM(
        by_schema={
            "FactList": [{"facts": []}, {"facts": [{"relation": "age", "value": "32 years old"}]}],
            "text": [invented, invented],
        }
    )
    record = await sanitise("d", SAMPLE_TEXT, llm)
    assert record.meta.leak_retries == 1
    assert "Arthur" not in record.rewriter.safe_text and "Penrose" not in record.rewriter.safe_text
    assert REDACTED in record.rewriter.safe_text
