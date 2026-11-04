"""The legal run end to end without a model: build, evaluate, report, export (slow: spaCy)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.presidio
async def test_mock_run_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("spacy")
    from sanitise.legal.export import export_bundle
    from sanitise.legal.report import write_report
    from sanitise.legal.run import RunConfig, build_corpus_store, evaluate
    from sanitise.mock_mode import mock_llm, mock_sanitise

    # the three targets' documents, so the attacks retrieve them
    data = tmp_path / "legal"
    shutil.copytree(REPO / "data" / "legal", data, ignore=shutil.ignore_patterns("train*"))
    corpus = data / "corpus"
    for path in corpus.glob("*.json"):
        if not path.name.startswith(("c001", "c002", "c003", "c004", "c005")):
            path.unlink()
    cfg = RunConfig(
        data_dir=data,
        store_path=tmp_path / "store.sqlite",
        out_dir=tmp_path / "results",
        retrieval_mode="bm25",
        mi_limit=5,
    )
    report = await build_corpus_store(cfg, mock_llm(), sanitiser=mock_sanitise)
    assert not report.failed and not report.leaky
    results = await evaluate(cfg, mock_llm())

    a1 = results["agents"]["agent1"]
    assert a1["direct_recall"] >= 0.95 and a1["precision"] >= 0.80
    assert results["pipelines"]["protected"]["pii_leakage"]["rate"] == 0.0
    assert results["pipelines"]["protected"]["extraction"]["exact_identifier_matches"] == 0
    for row in results["attack_outcomes"]:
        assert row["protected_identified_at"] is None, row
        if row["kind"] in {"direct", "linkage"}:
            assert row["baseline_identified_at"] is not None, row  # the baseline leaks
    stepwise = next(
        r
        for r in results["attack_outcomes"]
        if r["attack_id"] == "stepwise_c001" and r["monitor"] == "monitor"
    )
    assert stepwise["blocked_rounds"][0] == 2  # "which one involves a doctor?"

    assert (write_report(results, cfg.out_dir)).is_file()
    bundle = export_bundle(data, cfg.store_path, cfg.out_dir, tmp_path / "bundle")
    recorded = json.loads((bundle / "recorded_calls.json").read_text(encoding="utf-8"))
    assert recorded["llm"] and recorded["retrieval"]
