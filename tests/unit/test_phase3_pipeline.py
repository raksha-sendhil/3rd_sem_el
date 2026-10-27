"""The offline path (Agents 1 -> 2 -> 3) with MockLLM: orchestration, leak retry, fallback."""

from __future__ import annotations

import pytest

from sanitise.agents.extractor.prompt import extract_facts
from sanitise.agents.pipeline import sanitise
from sanitise.agents.rewriter.leakcheck import leak_check, mask_fallback
from sanitise.agents.rewriter.ontology import generalise
from sanitise.agents.rewriter.pipeline import apply_decisions, plan_decisions, removed_decisions
from sanitise.agents.rewriter.placeholders import allocate_placeholders
from sanitise.agents.rewriter.routing import route
from sanitise.errors import SchemaValidationFailed
from sanitise.llm import MockLLM
from sanitise.schemas import Action, ExtractorOutput, PIICategory, SanitisedRecord
from tests.conftest import SAMPLE_TEXT, fake_rule_detect

C = PIICategory

FACTS = {
    "facts": [
        {
            "subject": "writer.son",
            "relation": "age",
            "value": "32 years old",
            "importance": "medium",
        },
        {
            "subject": "writer.son",
            "relation": "works_as",
            "value": "lighthouse keeper",
            "importance": "low",
        },
        {"subject": "nobody", "relation": "other", "value": "dropped", "importance": "low"},
    ]
}
CLEAN = "The writer's son, in their 30s, is tall and has had abdominal pain for ten weeks."


def _detected():
    from sanitise.schemas import DetectorOutput

    return DetectorOutput(doc_id="d", spans=fake_rule_detect(SAMPLE_TEXT))


async def test_sanitise_end_to_end(fake_rules: None) -> None:
    llm = MockLLM(by_schema={"FactList": [FACTS], "text": [CLEAN]})
    record = await sanitise("d", SAMPLE_TEXT, llm)
    assert isinstance(record, SanitisedRecord)
    assert record.rewriter.safe_text == CLEAN and record.residual_leaks == []
    assert {s.id for s in record.subjects} >= {"writer", "writer.son"}
    assert all(f.subject != "nobody" for f in record.extractor.facts)  # unknown subject dropped
    assert record.meta.leak_retries == 0
    assert len(record.rewriter.decisions) == len(record.detector.spans)
    phone = next(d for d in record.rewriter.decisions if d.category is C.PHONE)
    assert phone.action is Action.PLACEHOLDER and phone.replacement == "[PHONE_1]"


async def test_leak_is_retried_then_masked(fake_rules: None) -> None:
    leaky = "Call 07700 900123; he is 32 years old."
    llm = MockLLM(by_schema={"FactList": [FACTS], "text": [leaky, leaky]})
    record = await sanitise("d", SAMPLE_TEXT, llm)
    assert record.meta.leak_retries == 1
    assert "07700 900123" not in record.rewriter.safe_text
    assert "[PHONE_1]" in record.rewriter.safe_text and record.residual_leaks == []
    retry = [c for c in llm.calls if c.schema_name == "text"][1]
    assert "07700" not in retry.user  # the leaked text is never sent back to the model


async def test_reconstruction_prompt_never_contains_removed_text(fake_rules: None) -> None:
    llm = MockLLM(by_schema={"FactList": [FACTS], "text": [CLEAN]})
    await sanitise("d", SAMPLE_TEXT, llm)
    rewrite_call = next(c for c in llm.calls if c.schema_name == "text")
    assert "32 years old" not in rewrite_call.user and "in their 30s" in rewrite_call.user


async def test_bad_fact_output_is_a_schema_error(fake_rules: None) -> None:
    bad = {
        "facts": [
            {
                "subject": "writer",
                "relation": "favourite_colour",
                "value": "x",
                "importance": "high",
            }
        ]
    }
    llm = MockLLM(by_schema={"FactList": [bad] * 4}, retries=3)
    with pytest.raises(SchemaValidationFailed) as err:
        await sanitise("d", SAMPLE_TEXT, llm)
    assert err.value.stage == "extract"


async def test_extract_facts_returns_fact_tuples() -> None:
    llm = MockLLM(by_schema={"FactList": [FACTS]})
    facts = await extract_facts(SAMPLE_TEXT, [], [], llm)
    assert [f.relation for f in facts] == ["age", "works_as", "other"]


def test_routing_placeholders_and_apply() -> None:
    detected = _detected()
    decisions, tokens = plan_decisions(detected, ExtractorOutput(doc_id="d", facts=[]))
    assert len(decisions) == len(detected.spans)
    assert tokens == {"[PHONE_1]": "07700 900123"}
    safe = apply_decisions(SAMPLE_TEXT, decisions)
    assert "07700 900123" not in safe and "32 years old" not in safe and "My son" in safe
    assert leak_check(removed_decisions(decisions), safe) == []


def test_leak_check_and_mask_fallback() -> None:
    decisions = route(fake_rule_detect(SAMPLE_TEXT), [])
    tokens = allocate_placeholders(decisions)
    text = "Phone 07700 900123 or 07700 900123."
    leaks = leak_check(removed_decisions(decisions), text)
    assert len(leaks) == 2 and all(text[s.start : s.end] == s.text for s in leaks)
    assert mask_fallback(text, leaks, tokens) == "Phone [PHONE_1] or [PHONE_1]."


def test_generalise_age_bands() -> None:
    assert generalise("32 years old", C.AGE) == "in their 30s"
    assert generalise("6 years old", C.AGE) == "a child"  # Person A's ladder (was the placeholder)
