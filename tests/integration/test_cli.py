from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sanitise.cli.main import app
from sanitise.evaluation import scoring
from sanitise.schemas import DetectorResult, SpanMetrics
from tests.conftest import SAMPLE_QUASI, SAMPLE_RULE_HITS, SAMPLE_TEXT

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and "0.1.0" in result.output


def test_detect_prints_detector_result_json(fake_rules: None, fixture_file: Path) -> None:
    result = runner.invoke(app, ["detect", "--input", str(fixture_file), "--mock"])
    assert result.exit_code == 0, result.output
    parsed = DetectorResult.model_validate_json(result.stdout)
    assert parsed.meta.rule_span_count == len(SAMPLE_RULE_HITS)
    assert parsed.meta.model == "mock-llm"


def test_detect_verbose_layout(fake_rules: None, fixture_file: Path, tmp_path: Path) -> None:
    script = tmp_path / "mock.json"
    script.write_text(
        json.dumps({"QuasiList": [SAMPLE_QUASI.model_dump(mode="json")]}), encoding="utf-8"
    )
    result = runner.invoke(
        app, ["detect", "--input", str(fixture_file), "--verbose", "--mock-responses", str(script)]
    )
    assert result.exit_code == 0, result.output
    for heading in ("ORIGINAL DOCUMENT", "RULE PASS", "LLM PASS", "MERGED OUTPUT", "validation OK"):
        assert heading in result.output
    assert '"32 years old"' in result.output
    start = SAMPLE_TEXT.index("32 years old")
    assert f"chars {start}-{start + 12}" in result.output
    assert "lighthouse keeper + Port Isaac" in result.output
    assert f"{len(SAMPLE_RULE_HITS) + 2} spans" in result.output


def test_detect_rules_only(fake_rules: None, fixture_file: Path) -> None:
    result = runner.invoke(app, ["detect", "-i", str(fixture_file), "--rules-only", "--verbose"])
    assert result.exit_code == 0, result.output
    assert "skipped (--rules-only)" in result.output


def test_detect_batch_writes_outputs(fake_rules: None, tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    for i in range(3):
        (fixtures / f"case_{i:02d}.json").write_text(
            json.dumps({"text": f"{SAMPLE_TEXT} #{i}"}), encoding="utf-8"
        )
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["detect", "-i", str(fixtures), "--out", str(out), "--mock", "-j", "2"]
    )
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("*.json"))) == 3
    assert "3/3 documents processed" in result.output


def test_detect_doc_too_long_prints_envelope(tmp_path: Path) -> None:
    path = tmp_path / "long.txt"
    path.write_text("x" * 2001, encoding="utf-8")
    result = runner.invoke(app, ["detect", "-i", str(path), "--mock"])
    assert result.exit_code == 1
    assert "DOC_TOO_LONG" in result.output


def test_detect_llm_failure_exit_code(fake_rules: None, fixture_file: Path, tmp_path: Path) -> None:
    script = tmp_path / "mock.json"
    script.write_text(json.dumps({"QuasiList": ["not json"] * 4}), encoding="utf-8")
    result = runner.invoke(
        app, ["detect", "-i", str(fixture_file), "--mock-responses", str(script)]
    )
    assert result.exit_code == 1
    assert "SCHEMA_VALIDATION_FAILED" in result.output and "llm_pass" in result.output


def test_score_not_implemented_yet(
    gold_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_yet(pred, gold):
        raise NotImplementedError("score() is not implemented yet (Person D)")

    monkeypatch.setattr(scoring, "score", not_yet)
    pred = tmp_path / "pred"
    pred.mkdir()
    result = runner.invoke(app, ["score", "--gold", str(gold_dir), "--pred", str(pred)])
    assert result.exit_code == 3
    assert "not implemented" in result.output


def test_score_prints_metrics(
    fake_rules: None,
    gold_dir: Path,
    fixture_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "out"
    assert (
        runner.invoke(
            app, ["detect", "-i", str(fixture_file), "--mock", "--out", str(out)]
        ).exit_code
        == 0
    )

    seen = {}

    def fake_score(pred, gold):
        seen["pred"], seen["gold"] = pred, gold
        return SpanMetrics(
            precision=0.89,
            recall=0.84,
            f1=0.86,
            support=6,
            per_category={"AGE": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}},
        )

    monkeypatch.setattr(scoring, "score", fake_score)
    result = runner.invoke(app, ["score", "--gold", str(gold_dir), "--pred", str(out)])
    assert result.exit_code == 0, result.output
    assert "Precision 0.89" in result.output and "Recall 0.84" in result.output
    assert "(1 documents, 6 gold spans)" in result.output
    assert seen["pred"][0].doc_id == seen["gold"][0].doc_id

    as_json = runner.invoke(app, ["score", "--gold", str(gold_dir), "--pred", str(out), "--json"])
    assert SpanMetrics.model_validate_json(as_json.stdout).recall == 0.84


def test_score_errors_and_report_with_real_scorer(
    fake_rules: None, gold_dir: Path, fixture_file: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    runner.invoke(app, ["detect", "-i", str(fixture_file), "--rules-only", "--out", str(out)])
    report = tmp_path / "report.md"
    result = runner.invoke(
        app,
        [
            "score",
            "--gold",
            str(gold_dir),
            "--pred",
            str(out),
            "--errors",
            "--report",
            str(report),
            "--model",
            "rules-only",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Recall" in result.output
    assert "MISSED" in result.output or "No errors." in result.output
    text = report.read_text(encoding="utf-8")
    assert "**Model:** rules-only" in text and "## Per category" in text
