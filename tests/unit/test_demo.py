"""The notebook helpers: analyse() and the HTML rendering."""

from __future__ import annotations

import pytest

from sanitise.demo import (
    analyse,
    analyse_sync,
    comparison_markdown,
    error_html,
    highlight,
    summarise_run,
    to_html,
)
from sanitise.errors import DocTooLong
from sanitise.llm import MockLLM
from sanitise.schemas import PIICategory as C
from sanitise.schemas import SensitiveSpan
from tests.conftest import SAMPLE_QUASI, SAMPLE_TEXT


async def test_analyse_runs_the_full_pipeline(fake_rules: None) -> None:
    result, trace = await analyse(SAMPLE_TEXT, MockLLM(handler=lambda call: SAMPLE_QUASI))
    assert trace.llm_ran and result.meta.rule_span_count == 5
    assert any(s.category is C.QUASI_COMBO for s in result.output.spans)


async def test_analyse_rules_only_never_calls_a_model(fake_rules: None) -> None:
    result, trace = await analyse(SAMPLE_TEXT, rules_only=True)
    assert not trace.llm_ran and result.meta.llm_span_count == 0


async def test_analyse_sync_works_inside_a_running_loop(fake_rules: None) -> None:
    # This test itself runs inside an event loop, like a notebook cell does.
    result, _ = analyse_sync(SAMPLE_TEXT, llm=MockLLM(handler=lambda call: SAMPLE_QUASI))
    assert result.meta.rule_span_count == 5


async def test_analyse_rejects_long_documents() -> None:
    with pytest.raises(DocTooLong):
        await analyse("x" * 2001, rules_only=True)


def test_highlight_escapes_and_marks_spans() -> None:
    text = "Dr <b> Lee, 40 years old"
    spans = [
        SensitiveSpan(text="Lee", category=C.NAME, start=7, end=10, method="rule"),
        SensitiveSpan(text="40 years old", category=C.AGE, start=12, end=24, method="rule"),
    ]
    out = highlight(text, spans)
    assert "&lt;b&gt;" in out and "<b>" not in out
    assert out.count("<mark") == 2 and ">NAME</sup>" in out and ">AGE</sup>" in out


async def test_to_html_shows_spans_combos_and_meta(fake_rules: None) -> None:
    result, trace = await analyse(SAMPLE_TEXT, MockLLM(handler=lambda call: SAMPLE_QUASI))
    page = to_html(SAMPLE_TEXT, result, trace)
    assert "lighthouse keeper + Port Isaac" in page
    assert "32 years old" in page and "mock-llm" in page
    assert result.doc_id in page


def test_summarise_run_and_comparison_table(
    fake_rules: None, gold_dir, fixture_file, tmp_path
) -> None:
    from typer.testing import CliRunner

    from sanitise.cli.main import app

    out = tmp_path / "out"
    CliRunner().invoke(app, ["detect", "-i", str(fixture_file), "--mock", "--out", str(out)])
    row = summarise_run("mock", out, gold_dir)
    assert row["documents"] == 1 and 0 <= row["recall"] <= 1 and row["retries"] == 0
    table = comparison_markdown([row], gold_spans=6)
    assert "| mock | 1 |" in table and "6 gold spans" in table


def test_error_html_shows_the_envelope() -> None:
    page = error_html(DocTooLong("too long", stage="load"))
    assert "DOC_TOO_LONG" in page and "load" in page
