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
    Subject,
)

REPO = Path(__file__).resolve().parents[2]
PROMPTS = {
    "agent1": quasi.SYSTEM_PROMPT + quasi.TASK_REMINDER,
    "extract": extract_prompt.SYSTEM_PROMPT,
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
    assert len(blocks) == 3
    for block in blocks:
        extract_prompt.LLMFactList.model_validate_json(block)


def test_extraction_user_prompt_lists_subjects_and_hints() -> None:
    user = extract_prompt.build_user_prompt(
        "I am 30.", [Subject(id="writer", label="the writer")], []
    )
    assert "- writer: the writer" in user and "(none)" in user and user.endswith("OUTPUT:")


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
    assert "the writer - identifier: [PERSON_1]." in user  # longest span replaced first
    assert "the writer's wife - has condition: gout." in user
    assert "person 1 - identifier: [PERSON_2]." in user


def test_answer_prompt_is_neutral_and_contains_the_records() -> None:
    system, user = answer_prompt.build_answer_prompt("Who has gout?", [("d1", "Asha has gout.")])
    assert "[d1]\nAsha has gout." in user and user.endswith("ANSWER:")
    assert "privacy" not in system.lower() and "protect" not in system.lower()
