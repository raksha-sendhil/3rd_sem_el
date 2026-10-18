"""Contract tests for the seams in spec 8.2.

These run against whatever implementation is currently in the repository - the
Day-1 placeholders now, the real code once A, B and D land it - so a signature
or contract drift fails here on the day it happens, not at Day-5 integration.
"""

from __future__ import annotations

import inspect

import pytest

from sanitise.agents.detector import merge, quasi, rules
from sanitise.agents.detector.pipeline import detect
from sanitise.evaluation import scoring
from sanitise.llm import MockLLM
from sanitise.schemas import DetectorOutput, PIICategory, SensitiveSpan, SpanMetrics
from sanitise.validation import check_span_offsets
from tests.conftest import SAMPLE_QUASI, SAMPLE_TEXT, fake_rule_detect


def params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


def test_signatures_match_spec() -> None:
    assert params(rules.rule_detect) == ["text"]
    assert params(merge.merge_spans) == ["rule_spans", "llm_spans"]
    assert params(quasi.infer_quasi) == ["text", "explicit", "llm"]
    assert params(scoring.score) == ["pred", "gold"]
    assert params(detect) == ["doc_id", "text", "llm"]
    assert inspect.iscoroutinefunction(quasi.infer_quasi)
    assert inspect.iscoroutinefunction(detect)
    assert not inspect.iscoroutinefunction(rules.rule_detect)
    assert not inspect.iscoroutinefunction(merge.merge_spans)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("presidio_analyzer"),
    reason="presidio-analyzer not installed",
)
def test_rule_detect_contract() -> None:  # Person A
    spans = rules.rule_detect(SAMPLE_TEXT)
    assert isinstance(spans, list)
    for span in spans:
        assert isinstance(span, SensitiveSpan)
        assert span.method == "rule" and span.start is not None
    check_span_offsets(SAMPLE_TEXT, spans, stage="rule_pass")


async def test_infer_quasi_contract() -> None:  # Person B
    explicit = fake_rule_detect(SAMPLE_TEXT)
    spans = await quasi.infer_quasi(
        SAMPLE_TEXT, explicit, MockLLM(handler=lambda call: SAMPLE_QUASI)
    )
    assert spans, "infer_quasi dropped every hit the model returned"
    for span in spans:
        assert span.method == "llm" and span.rationale
        assert span.start is None or 0 <= span.start < span.end <= len(SAMPLE_TEXT)


async def test_infer_quasi_only_uses_the_client_it_is_given() -> None:  # Person B
    llm = MockLLM(handler=lambda call: SAMPLE_QUASI)
    await quasi.infer_quasi(SAMPLE_TEXT, [], llm)
    assert len(llm.calls) >= 1
    assert all(call.schema_name for call in llm.calls)


def test_merge_spans_contract() -> None:  # Person A
    rule_spans = fake_rule_detect(SAMPLE_TEXT)
    llm_spans = [
        SensitiveSpan(text=h.text, category=h.category, method="llm", rationale=h.rationale)
        for h in SAMPLE_QUASI.quasi_identifiers
    ]
    merged = merge.merge_spans(rule_spans, llm_spans)
    assert isinstance(merged, list) and all(isinstance(s, SensitiveSpan) for s in merged)
    check_span_offsets(SAMPLE_TEXT, merged, stage="merge")
    # Nothing explicit may be lost: every rule span's characters stay covered.
    for span in rule_spans:
        assert any(
            m.start is not None and m.start <= span.start and span.end <= m.end for m in merged
        ), f"merge dropped {span.text!r}"
    assert merge.merge_spans([], []) == []


def test_score_contract() -> None:  # Person D
    gold = [DetectorOutput(doc_id="d", spans=fake_rule_detect(SAMPLE_TEXT))]
    try:
        metrics = scoring.score(gold, gold)
    except NotImplementedError:
        pytest.skip("score() not implemented yet (Person D)")
    assert isinstance(metrics, SpanMetrics)
    assert metrics.precision == metrics.recall == metrics.f1 == 1.0
    assert metrics.support == len(gold[0].spans)
    assert set(metrics.per_category) <= {c.value for c in PIICategory}
