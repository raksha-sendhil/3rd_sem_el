"""Person A's Agent 3 logic on the 20 fixtures and their gold spans (plan 10.5, Person A, 3-4).

* every span gets exactly one decision, in order, offsets copied;
* every placeholder / generalised span is gone from ``apply_decisions(text, decisions)``, and
  the leak check finds nothing there;
* if the model copies the original text back on both attempts, the deterministic mask leaves
  ``rewrite`` clean - no ``LEAK_CHECK_FAILED`` and no residual leak;
* with the real rule pass (``-m presidio``), ``sanitise()`` with the mock-mode model gives
  ``residual_leaks == []`` for every fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sanitise.agents.extractor.canonical import subjects
from sanitise.agents.rewriter.leakcheck import leak_check
from sanitise.agents.rewriter.pipeline import (
    apply_decisions,
    plan_decisions,
    removed_decisions,
    rewrite_with_trace,
)
from sanitise.llm import MockLLM
from sanitise.results import load_gold
from sanitise.schemas import Action, DetectorOutput, ExtractorOutput, FactTuple, Importance

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = sorted((ROOT / "data" / "fixtures").glob("*.json"))
GOLD = {g.doc_id: g for g in load_gold(ROOT / "data" / "gold")}


def fixture(path: Path) -> tuple[str, str, DetectorOutput]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc["doc_id"], doc["text"], GOLD[doc["doc_id"]]


def test_there_are_twenty_fixtures() -> None:
    assert len(FIXTURES) == 20


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_decisions_cover_every_span_and_remove_it(path: Path) -> None:
    doc_id, text, gold = fixture(path)
    decisions, tokens = plan_decisions(gold, ExtractorOutput(doc_id=doc_id, facts=[]))
    assert len(decisions) == len(gold.spans)
    for span, d in zip(gold.spans, decisions, strict=True):
        assert (d.span_text, d.category, d.start, d.end) == (
            span.text,
            span.category,
            span.start,
            span.end,
        )
        if d.action is not Action.KEEP:
            assert d.replacement, f"{d.span_text!r} has no replacement"
    assert set(tokens) == {d.replacement for d in decisions if d.action is Action.PLACEHOLDER}
    safe = apply_decisions(text, decisions)
    for d in removed_decisions(decisions):
        assert d.span_text not in safe
    assert leak_check(removed_decisions(decisions), safe) == []


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
async def test_model_that_copies_the_original_is_masked_clean(path: Path) -> None:
    doc_id, text, gold = fixture(path)
    llm = MockLLM(by_schema={"text": [text, text]})
    # one fact, so Agent 3 calls the model (with no facts it writes a fixed sentence instead)
    unwell = FactTuple(
        subject="writer", relation="other", value="unwell", importance=Importance.LOW
    )
    trace = await rewrite_with_trace(
        doc_id, gold, ExtractorOutput(doc_id=doc_id, facts=[unwell]), llm
    )
    assert trace.residual_leaks == []
    removed = removed_decisions(trace.output.decisions)
    assert leak_check(removed, trace.output.safe_text) == []
    if removed:
        assert trace.leak_retries == 1


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_subjects_mentions_come_from_the_text(path: Path) -> None:
    _, text, gold = fixture(path)
    subs = subjects(text, gold.spans)
    assert subs[0].id == "writer"
    assert len({s.id for s in subs}) == len(subs)
    for s in subs:
        assert all(m in text for m in s.mentions)


@pytest.mark.presidio
async def test_sanitise_with_mock_model_leaves_no_residual_leaks() -> None:
    from sanitise.agents.pipeline import sanitise
    from sanitise.mock_mode import mock_llm

    for path in FIXTURES:
        doc = json.loads(path.read_text(encoding="utf-8"))
        record = await sanitise(doc["doc_id"], doc["text"], mock_llm())
        assert record.residual_leaks == [], path.stem
