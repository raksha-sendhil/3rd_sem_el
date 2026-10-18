"""detect() orchestration, with each stage faked so the wiring is tested on its own."""

from __future__ import annotations

import asyncio

import pytest

from sanitise.agents.detector.pipeline import (
    DetectorStages,
    detect,
    detect_llm,
    detect_rules,
    detect_with_trace,
)
from sanitise.agents.detector.quasi import infer_quasi
from sanitise.errors import DocTooLong, LLMTimeout, OffsetOutOfRange, SchemaValidationFailed
from sanitise.llm import MockLLM
from sanitise.loader import sha256_hex
from sanitise.schemas import PIICategory, SensitiveSpan
from tests.conftest import SAMPLE_QUASI, SAMPLE_RULE_HITS, fake_rule_detect


def union(rule: list[SensitiveSpan], llm: list[SensitiveSpan]) -> list[SensitiveSpan]:
    return [*rule, *llm]


STAGES = DetectorStages(rule_detect=fake_rule_detect, infer_quasi=infer_quasi, merge_spans=union)


async def test_full_pipeline_counts_and_meta(
    sample_text: str, sample_doc_id: str, mock_llm: MockLLM
) -> None:
    result, trace = await detect_with_trace(sample_doc_id, sample_text, mock_llm, stages=STAGES)
    assert result.doc_id == result.output.doc_id == sample_doc_id
    assert result.original_sha256 == sha256_hex(sample_text)
    meta = result.meta
    assert meta.model == "mock-llm"
    assert meta.rule_span_count == len(SAMPLE_RULE_HITS) == len(trace.rule_spans)
    assert meta.llm_span_count == len(SAMPLE_QUASI.quasi_identifiers) == len(trace.llm_spans)
    assert len(result.output.spans) == meta.rule_span_count + meta.llm_span_count
    assert meta.retries == 0
    assert meta.total_ms >= meta.rule_ms + meta.llm_ms - 1
    assert all(s.method == "llm" and s.start is None and s.rationale for s in trace.llm_spans)


async def test_detect_uses_default_stages(
    fake_rules: None, sample_text: str, mock_llm: MockLLM
) -> None:
    result = await detect("doc", sample_text, mock_llm)
    assert result.meta.rule_span_count == len(SAMPLE_RULE_HITS)


async def test_retries_are_reported(sample_text: str) -> None:
    llm = MockLLM(
        [
            '{"quasi_identifiers": [{"text": "x", "category": "NOPE", "rationale": "r"}]}',
            SAMPLE_QUASI,
        ]
    )
    result = await detect_with_trace("d", sample_text, llm, stages=STAGES)
    assert result[0].meta.retries == 1


async def test_explicit_spans_reach_the_llm_prompt(sample_text: str, mock_llm: MockLLM) -> None:
    await detect_with_trace("d", sample_text, mock_llm, stages=STAGES)
    assert "07700 900123" in mock_llm.calls[0].user


async def test_rules_only_skips_llm(sample_text: str, mock_llm: MockLLM) -> None:
    result, trace = await detect_with_trace(
        "d", sample_text, mock_llm, run_llm=False, stages=STAGES
    )
    assert mock_llm.calls == [] and not trace.llm_ran
    assert (result.meta.model, result.meta.llm_ms, result.meta.llm_span_count) == ("none", 0, 0)


async def test_doc_too_long(mock_llm: MockLLM) -> None:
    with pytest.raises(DocTooLong) as info:
        await detect_with_trace("d", "x" * 2001, mock_llm, stages=STAGES)
    assert info.value.stage == "load"


async def test_llm_failure_is_tagged_with_stage(sample_text: str) -> None:
    with pytest.raises(LLMTimeout) as info:
        await detect_with_trace("d", sample_text, MockLLM([LLMTimeout("slow")]), stages=STAGES)
    assert info.value.stage == "llm_pass"

    with pytest.raises(SchemaValidationFailed) as info2:
        await detect_with_trace("d", sample_text, MockLLM(["garbage"] * 4), stages=STAGES)
    assert (info2.value.stage, info2.value.attempts) == ("llm_pass", 4)


@pytest.mark.parametrize(
    ("bad_span", "error"),
    [
        (
            SensitiveSpan(text="My son", category="RELATION", start=0, end=999, method="rule"),
            OffsetOutOfRange,
        ),
        (
            SensitiveSpan(text="Your daughter", category="RELATION", start=0, end=6, method="rule"),
            OffsetOutOfRange,
        ),
        (SensitiveSpan(text="My son", category="RELATION", method="rule"), OffsetOutOfRange),
        (
            SensitiveSpan(
                text="My son", category="RELATION", start=0, end=6, method="llm", rationale="r"
            ),
            SchemaValidationFailed,
        ),
    ],
)
async def test_rule_pass_contract_enforced(
    sample_text: str, mock_llm: MockLLM, bad_span: SensitiveSpan, error: type
) -> None:
    stages = DetectorStages(
        rule_detect=lambda text: [bad_span], infer_quasi=infer_quasi, merge_spans=union
    )
    with pytest.raises(error) as info:
        await detect_with_trace("d", sample_text, mock_llm, stages=stages)
    assert info.value.stage == "rule_pass"


async def test_rule_span_case_difference_is_accepted(sample_text: str, mock_llm: MockLLM) -> None:
    # The spec's own example labels "My son" at 0-6 as "my son".
    span = SensitiveSpan(text="my son", category="RELATION", start=0, end=6, method="rule")
    stages = DetectorStages(
        rule_detect=lambda text: [span], infer_quasi=infer_quasi, merge_spans=union
    )
    result, _ = await detect_with_trace("d", sample_text, mock_llm, stages=stages)
    assert result.output.spans[0].text == "my son"


async def test_mutated_span_is_revalidated(sample_text: str, mock_llm: MockLLM) -> None:
    def sneaky(text: str) -> list[SensitiveSpan]:
        span = SensitiveSpan(text="My son", category="RELATION", start=0, end=6, method="rule")
        span.category = "DIAGNOSIS"  # type: ignore[assignment]  # no validate_assignment
        return [span]

    stages = DetectorStages(rule_detect=sneaky, infer_quasi=infer_quasi, merge_spans=union)
    with pytest.raises(SchemaValidationFailed, match="category"):
        await detect_with_trace("d", sample_text, mock_llm, stages=stages)


async def test_merge_output_is_validated(sample_text: str, mock_llm: MockLLM) -> None:
    def broken_merge(rule: list[SensitiveSpan], llm: list[SensitiveSpan]) -> list[SensitiveSpan]:
        return [SensitiveSpan(text="nowhere", category="NAME", start=5000, end=5007, method="rule")]

    stages = DetectorStages(
        rule_detect=fake_rule_detect, infer_quasi=infer_quasi, merge_spans=broken_merge
    )
    with pytest.raises(OffsetOutOfRange) as info:
        await detect_with_trace("d", sample_text, mock_llm, stages=stages)
    assert info.value.stage == "merge"


async def test_rule_pass_runs_off_the_event_loop(sample_text: str, mock_llm: MockLLM) -> None:
    loop_thread_ids: list[bool] = []

    def probe(text: str) -> list[SensitiveSpan]:
        try:
            asyncio.get_running_loop()
            loop_thread_ids.append(True)
        except RuntimeError:
            loop_thread_ids.append(False)
        return []

    stages = DetectorStages(rule_detect=probe, infer_quasi=infer_quasi, merge_spans=union)
    await detect_with_trace("d", sample_text, mock_llm, stages=stages)
    assert loop_thread_ids == [False]


async def test_concurrent_detects_keep_separate_retry_counts(sample_text: str) -> None:
    bad = '{"quasi_identifiers": [{"text": "x", "category": "NOPE", "rationale": "r"}]}'
    flaky = MockLLM([bad, bad, SAMPLE_QUASI], delay_s=0.01)
    clean = MockLLM([SAMPLE_QUASI], delay_s=0.01)
    (r1, _), (r2, _) = await asyncio.gather(
        detect_with_trace("a", sample_text, flaky, stages=STAGES),
        detect_with_trace("b", sample_text, clean, stages=STAGES),
    )
    assert (r1.meta.retries, r2.meta.retries) == (2, 0)


async def test_single_pass_helpers(sample_text: str, mock_llm: MockLLM) -> None:
    rules_out = await detect_rules("d", sample_text, stages=STAGES)
    assert {s.method for s in rules_out.spans} == {"rule"}
    llm_out = await detect_llm("d", sample_text, rules_out.spans, mock_llm, stages=STAGES)
    assert {s.category for s in llm_out.spans} == {PIICategory.QUASI_COMBO}

    bad_explicit = [SensitiveSpan(text="zzz", category="NAME", start=0, end=3, method="rule")]
    with pytest.raises(OffsetOutOfRange):
        await detect_llm("d", sample_text, bad_explicit, mock_llm, stages=STAGES)
