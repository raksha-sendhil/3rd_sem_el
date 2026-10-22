"""The advisor-demo views: one-system presentation, scores card, dataset summary."""

from __future__ import annotations

import re
from pathlib import Path

from sanitise.agents.detector.pipeline import detect
from sanitise.errors import DocTooLong, LLMUnreachable
from sanitise.evaluation import scoring
from sanitise.llm import MockLLM
from sanitise.results import load_gold, load_predictions
from sanitise.showcase import (
    dataset_summary,
    render_document,
    render_error,
    render_scores,
    run_fixtures,
)
from tests.conftest import SAMPLE_QUASI, SAMPLE_TEXT

REPO = Path(__file__).resolve().parents[2]
SOURCE_WORDS = re.compile(r"\b(rules?|llm|llama|presidio|spacy|regex|mock|model)\b", re.I)


async def test_render_document_never_says_where_a_span_came_from(fake_rules: None) -> None:
    result = await detect("d", SAMPLE_TEXT, MockLLM(handler=lambda call: SAMPLE_QUASI))
    page = render_document(SAMPLE_TEXT, result)
    visible = re.sub(r"<[^>]+>", " ", page)  # text the viewer sees, tooltips included below
    tooltips = " ".join(re.findall(r'title="([^"]*)"', page))
    assert not SOURCE_WORDS.search(visible + " " + tooltips), SOURCE_WORDS.findall(visible)
    assert "dashed" not in page  # every span is drawn the same way
    assert "32 years old" in page and "lighthouse keeper + Port Isaac" in page
    assert f"<b>{len(result.output.spans)}</b> sensitive items" in page


def test_render_document_without_spans() -> None:
    from sanitise.schemas import DetectorMeta, DetectorOutput, DetectorResult

    meta = DetectorMeta(
        model="m", rule_ms=1, llm_ms=1, total_ms=2, retries=0, rule_span_count=0, llm_span_count=0
    )
    result = DetectorResult(
        doc_id="d", original_sha256="x", output=DetectorOutput(doc_id="d", spans=[]), meta=meta
    )
    page = render_document("plain text", result)
    assert "Nothing sensitive found." in page and "None found." in page


def test_dataset_summary_matches_the_gold_set() -> None:
    d = dataset_summary(REPO / "data/fixtures", REPO / "data/gold")
    assert d == {
        "documents": 20,
        "patient_questions": 16,
        "emails": 4,
        "gold_spans": 124,
        "combinations": 12,
        "nothing_sensitive": 3,
    }


async def test_run_fixtures_then_render_scores(fake_rules: None, tmp_path: Path) -> None:
    out = tmp_path / "out"
    results = await run_fixtures(
        REPO / "data/fixtures", out, MockLLM(handler=lambda call: SAMPLE_QUASI)
    )
    assert len(results) == 20 and len(list(out.glob("*.json"))) == 20
    metrics = scoring.score(load_predictions(out), load_gold(REPO / "data/gold"))
    page = render_scores(metrics, dataset_summary(REPO / "data/fixtures", REPO / "data/gold"))
    assert "Recall" in page and "124 hand-labelled" in page and "rules + llama3:8b" in page
    assert f"{metrics.recall:.2f}" in page


def test_render_error_is_plain_and_hides_stages() -> None:
    long_doc = render_error(
        DocTooLong("document has 2400 characters; the limit is 2000", stage="load")
    )
    assert "DOC_TOO_LONG" in long_doc and "2400 characters" in long_doc
    down = render_error(LLMUnreachable("cannot reach Ollama", stage="llm_pass"))
    assert "Run cell 1" in down and "llm_pass" not in down and "Ollama" not in down
