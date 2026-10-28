"""The LLM prompts: no evaluation data inside them, well-formed examples, safe inputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sanitise.agents.detector import quasi
from sanitise.agents.extractor import prompt as extract_prompt
from sanitise.agents.rewriter import prompt as rewrite_prompt
from sanitise.rag import answer_prompt
from sanitise.schemas import (
    Action,
    FactTuple,
    Importance,
    PIICategory,
    RoutingDecision,
    SensitiveSpan,
    Subject,
)

REPO = Path(__file__).resolve().parents[2]
PROMPTS = {
    "agent1": quasi.SYSTEM_PROMPT + quasi.TASK_REMINDER,
    "extract": extract_prompt.SYSTEM_PROMPT,
    "extract_document": extract_prompt.DOCUMENT_PROMPT,
    "rewrite": rewrite_prompt.SYSTEM_PROMPT,
    "answer": answer_prompt.SYSTEM_PROMPT,
}
COMMON = {"my son", "my wife", "my daughter", "my husband", "my mother", "my father", "my sister"}


def _gold() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in (REPO / "data/gold").glob("*.json")]


@pytest.mark.parametrize("name", list(PROMPTS))
def test_no_gold_text_in_any_prompt(name: str) -> None:
    """The evaluation runs on data/gold; a gold phrase in a prompt would inflate the scores."""
    prompt = PROMPTS[name].lower()
    leaks = set()
    for doc in _gold():
        for span in doc["spans"]:
            text = span["text"].lower()
            single_word_identifier = span["category"] in {
                "NAME",
                "LOCATION",
                "OCCUPATION",
                "MEDICAL_CONDITION",
            }
            if text in COMMON or span["start"] is None:
                continue
            if (len(text.split()) >= 2 or single_word_identifier) and re.search(
                rf"\b{re.escape(text)}\b", prompt
            ):
                leaks.add(text)
        for sentence in re.split(r"(?<=[.?!])\s+", doc["text"].lower()):
            if len(sentence) > 30 and sentence in prompt:
                leaks.add(sentence)
    assert not leaks, f"{name} prompt contains gold text: {sorted(leaks)}"


def test_extraction_examples_parse_with_the_decoding_schema() -> None:
    blocks = re.findall(r"OUTPUT:\n(\{.*?\]\})", extract_prompt.SYSTEM_PROMPT, re.S)
    assert len(blocks) == 4
    for block in blocks:
        extract_prompt.LLMFactList.model_validate_json(block)


def test_extraction_user_prompt_names_one_person_and_no_identifier_hints() -> None:
    spans = [
        SensitiveSpan(text="30", category=PIICategory.AGE, start=5, end=7, method="rule"),
        SensitiveSpan(text="Ira Sen", category=PIICategory.NAME, start=0, end=7, method="rule"),
    ]
    son = Subject(id="writer.son", label="the writer's son", mentions=["My son", "he"])
    user = extract_prompt.build_user_prompt("I am 30.", son, spans)
    assert "PERSON: writer.son (the writer's son), called: My son, he" in user
    assert "- AGE: 30" in user and "Ira Sen" not in user and user.endswith("OUTPUT:")


def test_the_model_is_never_asked_for_identifiers() -> None:
    schema = json.dumps(extract_prompt.LLMFactList.model_json_schema())
    for relation in ("identifier", "contact", "relation_to", "subject", "importance"):
        assert f'"{relation}"' not in schema


def _decision(text: str, cat: PIICategory, action: Action, repl: str | None) -> RoutingDecision:
    return RoutingDecision(
        span_text=text, category=cat, action=action, replacement=repl, reason="test"
    )


def test_reconstruction_input_has_no_removed_text_and_labels_subjects() -> None:
    facts = [
        FactTuple(
            subject="writer", relation="identifier", value="Asha Varma", importance=Importance.LOW
        ),
        FactTuple(
            subject="writer.wife",
            relation="has_condition",
            value="gout",
            importance=Importance.HIGH,
        ),
        FactTuple(
            subject="person_1", relation="identifier", value="Asha", importance=Importance.LOW
        ),
    ]
    decisions = [
        _decision("Asha", PIICategory.NAME, Action.PLACEHOLDER, "[PERSON_2]"),
        _decision("Asha Varma", PIICategory.NAME, Action.PLACEHOLDER, "[PERSON_1]"),
        _decision("gout", PIICategory.MEDICAL_CONDITION, Action.KEEP, None),
    ]
    user = rewrite_prompt.build_user_prompt(facts, decisions)
    assert "Asha" not in user
    assert "the writer - name: [PERSON_1]." in user  # longest span replaced first
    assert "the writer's wife - has condition: gout." in user
    assert "person 1 (not the writer) - name: [PERSON_2]." in user


def test_answer_prompt_is_neutral_and_contains_the_records() -> None:
    system, user = answer_prompt.build_answer_prompt("Who has gout?", [("d1", "Asha has gout.")])
    assert "[d1]\nAsha has gout." in user and user.endswith("ANSWER:")
    assert "privacy" not in system.lower() and "protect" not in system.lower()


def _fact(subject: str, relation: str, value: str) -> FactTuple:
    return FactTuple(subject=subject, relation=relation, value=value, importance=Importance.LOW)


def test_reconstruction_input_replaces_other_forms_of_a_removed_detail() -> None:
    decisions = [
        _decision("Ruby Hall Clinic, Wakad", PIICategory.LOCATION, Action.GENERALISE, "a clinic"),
        _decision("Omar Siddiqui", PIICategory.NAME, Action.PLACEHOLDER, "[PERSON_1]"),
    ]
    user = rewrite_prompt.build_user_prompt(
        [_fact("writer", "lives_in", "Wakad"), _fact("writer", "other", "signed Omar")], decisions
    )
    assert "Wakad" not in user and "Omar" not in user
    assert "lives in: a clinic." not in user  # the part of the place gets its own phrase
    assert "other: signed [PERSON_1]." in user


def test_reconstruction_input_drops_identifiers_that_are_not_tokens() -> None:
    decisions = [
        _decision("HC-3381925", PIICategory.ID_NUMBER, Action.PLACEHOLDER, "[ID_1]"),
        _decision("98111 20345", PIICategory.PHONE, Action.PLACEHOLDER, "[PHONE_1]"),
    ]
    facts = [
        _fact("writer", "identifier", "the writer"),  # not an identifier at all
        _fact("writer", "identifier", "Nora Quint"),  # a name Agent 1 missed
        _fact("writer.husband", "identifier", "HC-338192525"),  # the ID, copied wrongly
        _fact("writer", "contact", "Let me know"),
        _fact("writer", "contact", "98111 20345"),  # becomes [PHONE_1]: kept
        _fact("writer.husband", "relation_to", "husband"),  # a relative: kept
        _fact("person_1", "relation_to", "Nora"),  # not the writer's relative
    ]
    user = rewrite_prompt.build_user_prompt(facts, decisions)
    for gone in ("the writer - identifier", "Nora", "HC-33819", "Let me know"):
        assert gone not in user
    assert "the writer - contact: [PHONE_1]." in user
    assert "relation to" not in user  # the label says it: "the writer's husband"
