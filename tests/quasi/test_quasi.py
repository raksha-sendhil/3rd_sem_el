"""Person B's LLM pass: prompt, constrained schema, grounding and post-processing.

All against MockLLM - no model needed. Live behaviour is covered by
tests/integration/test_ollama_live.py (``pytest -m ollama``).
"""

from __future__ import annotations

import json

import pytest

from sanitise.agents.detector import quasi
from sanitise.agents.detector.pipeline import detect
from sanitise.agents.detector.quasi import LLMQuasiHit, build_user_prompt, ground, to_spans
from sanitise.errors import SchemaValidationFailed
from sanitise.llm import MockLLM
from sanitise.schemas import PIICategory as C
from tests.conftest import SAMPLE_TEXT, fake_rule_detect

TEXT = (
    "I am a lighthouse keeper on Lundy. My doctor at North Devon Hospital thinks my rash is "
    "contact dermatitis. Contact Dermatitis runs in the family."
)


def hit(text: str, category: str, rationale: str = "because") -> LLMQuasiHit:
    return LLMQuasiHit(text=text, category=category, rationale=rationale)


# ------------------------------------------------------------------ prompt


def test_user_prompt_lists_document_and_explicit_spans() -> None:
    explicit = fake_rule_detect(SAMPLE_TEXT)
    prompt = build_user_prompt(SAMPLE_TEXT, explicit)
    assert SAMPLE_TEXT in prompt
    assert "- AGE: 32 years old" in prompt and "- RELATION: My son" in prompt
    assert build_user_prompt("x", []).count("(none)") == 1


def test_schema_sent_to_the_model_is_narrowed_but_keeps_its_name() -> None:
    llm = MockLLM()
    import asyncio

    asyncio.run(quasi.infer_quasi(TEXT, [], llm))
    call = llm.calls[0]
    assert call.schema_name == "QuasiList"  # MockLLM scripts keyed on QuasiList still work
    enum = call.json_schema["properties"]["quasi_identifiers"]["items"]["properties"]["category"]
    assert set(enum["enum"]) == {
        "NAME",
        "AGE",
        "MEDICAL_CONDITION",
        "OCCUPATION",
        "LOCATION",
        "QUASI_COMBO",
    }
    assert "EXAMPLE 1" in call.system


# ------------------------------------------------------------------ grounding


def test_ground_finds_every_occurrence_case_insensitively() -> None:
    first, second = TEXT.index("contact dermatitis"), TEXT.index("Contact Dermatitis")
    assert ground(TEXT, "contact dermatitis") == [(first, first + 18), (second, second + 18)]


def test_ground_tolerates_articles_punctuation_and_whitespace() -> None:
    keeper, lundy = TEXT.index("lighthouse keeper"), TEXT.index("Lundy")
    assert ground(TEXT, "a lighthouse keeper") == [(keeper, keeper + 17)]
    assert ground(TEXT, '"Lundy."') == [(lundy, lundy + 5)]
    assert ground("works as a tea\n  taster now", "tea taster") == [(11, 23)]


def test_ground_matches_whole_words_only() -> None:
    assert ground("my heart rate", "art") == []
    assert ground(TEXT, "dermatitis in the elbow") == []


def test_to_spans_grounds_details_and_keeps_combos_offsetless() -> None:
    spans = to_spans(
        TEXT,
        [
            hit("lighthouse keeper", "OCCUPATION"),
            hit("North Devon Hospital", "LOCATION"),
            hit("contact dermatitis", "MEDICAL_CONDITION"),
            hit("lighthouse keeper + Lundy + dermatitis", "QUASI_COMBO"),
        ],
    )
    by_cat = {}
    for s in spans:
        by_cat.setdefault(s.category, []).append(s)
    assert [s.text for s in by_cat[C.OCCUPATION]] == ["lighthouse keeper"]
    assert [s.text for s in by_cat[C.MEDICAL_CONDITION]] == [
        "contact dermatitis",
        "Contact Dermatitis",
    ]
    combo = by_cat[C.QUASI_COMBO][0]
    assert combo.start is None and combo.end is None and combo.rationale == "because"
    for s in spans:
        assert s.method == "llm"
        if s.start is not None:
            assert TEXT[s.start : s.end] == s.text


def test_to_spans_drops_hallucinated_details() -> None:
    spans = to_spans(TEXT, [hit("asthma", "MEDICAL_CONDITION"), hit("Lundy", "LOCATION")])
    assert [s.text for s in spans] == ["Lundy"]


def test_to_spans_deduplicates_and_caps_combos() -> None:
    combos = [hit(f"combo {i}", "QUASI_COMBO") for i in range(5)]
    spans = to_spans(TEXT, [hit("Combo  0", "QUASI_COMBO"), *combos, hit("Lundy", "LOCATION")])
    assert sum(s.category is C.QUASI_COMBO for s in spans) == quasi.MAX_COMBOS
    assert sum(s.text == "Lundy" for s in spans) == 1


# ------------------------------------------------------------------ through the client


async def test_disallowed_category_is_repaired_by_the_retry_loop() -> None:
    bad = {"quasi_identifiers": [{"text": "rash", "category": "DATE", "rationale": "r"}]}
    good = {"quasi_identifiers": [{"text": "Lundy", "category": "LOCATION", "rationale": "r"}]}
    llm = MockLLM(by_schema={"QuasiList": [bad, good]})
    spans = await quasi.infer_quasi(TEXT, [], llm)
    assert [s.text for s in spans] == ["Lundy"]
    assert len(llm.calls) == 2 and llm.calls[1].is_repair
    assert "category" in llm.calls[1].messages[-1]["content"]


async def test_gives_up_after_the_retry_budget() -> None:
    llm = MockLLM(by_schema={"QuasiList": ["not json"] * 4}, retries=3)
    with pytest.raises(SchemaValidationFailed):
        await quasi.infer_quasi(TEXT, [], llm)
    assert len(llm.calls) == 4


async def test_detect_merges_grounded_llm_spans_with_rule_spans(
    fake_rules: None,
) -> None:
    reply = {
        "quasi_identifiers": [
            {"text": "lighthouse keeper", "category": "OCCUPATION", "rationale": "rare job"},
            # restates an explicit identifier: dropped by merge_spans
            {"text": "32 years old", "category": "OCCUPATION", "rationale": "wrong"},
            {
                "text": "lighthouse keeper + Port Isaac",
                "category": "QUASI_COMBO",
                "rationale": "rare job in a small village",
            },
        ]
    }
    llm = MockLLM(by_schema={"QuasiList": [json.dumps(reply)]})
    result = await detect("d", SAMPLE_TEXT, llm)
    spans = result.output.spans
    texts = {(s.category, s.text, s.method) for s in spans}
    assert (C.OCCUPATION, "lighthouse keeper", "llm") in texts
    assert (C.AGE, "32 years old", "rule") in texts
    assert not any(s.category is C.OCCUPATION and s.text == "32 years old" for s in spans)
    assert spans[-1].category is C.QUASI_COMBO and spans[-1].start is None
    assert result.meta.llm_span_count == 3  # counted before merge drops the restatement


async def test_names_and_ages_the_rules_missed_are_kept() -> None:
    text = "My daughter has a fever. She is Sunaina and she is eleven."
    reply = {
        "quasi_identifiers": [
            {"text": "Sunaina", "category": "NAME", "rationale": "The daughter's name."},
            {"text": "eleven", "category": "AGE", "rationale": "The daughter's age."},
        ]
    }
    spans = await quasi.infer_quasi(text, [], MockLLM(by_schema={"QuasiList": [reply]}))
    assert [(s.category, text[s.start : s.end]) for s in spans] == [
        (C.NAME, "Sunaina"),
        (C.AGE, "eleven"),
    ]
