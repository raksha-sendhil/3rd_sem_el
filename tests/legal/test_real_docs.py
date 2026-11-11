"""The agents-only path on real documents: data/legal/real, sanitise.legal.real_docs and the
Kaggle notebook that uses them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sanitise.legal import real_docs
from sanitise.legal.real_docs import (
    from_text,
    load_inputs,
    load_manifest,
    render_html,
    run_agents,
    split_text,
    summary_rows,
    totals,
    write_report,
)

REPO = Path(__file__).resolve().parents[2]
REAL = REPO / "data" / "legal" / "real"


def _saved(folder: Path) -> set[str]:
    return {p.name for p in folder.glob("*.json")}


def test_every_real_document_is_listed_with_its_source() -> None:
    docs = load_manifest(REAL)
    assert len(docs) >= 12
    assert sum(d.meta["jurisdiction"].startswith("India") for d in docs) >= 8
    assert len({d.meta["source"] for d in docs}) >= 5  # several sources, not one
    on_disk = {p.stem for p in (REAL / "docs").glob("*.txt")}
    assert on_disk == {d.id for d in docs}
    for d in docs:
        assert d.meta["url"].startswith("https://") and d.meta["licence"]
        assert d.must_hide and d.must_keep, d.id


def test_every_spot_check_string_is_in_its_document() -> None:
    for d in load_manifest(REAL):
        text = d.text.lower()
        missing = [s for s in d.must_hide + d.must_keep if s.lower() not in text]
        assert not missing, (d.id, missing)


def test_split_keeps_every_word_and_respects_the_limit() -> None:
    text = "\n\n".join(
        [
            "Short opening paragraph.",
            "A line.\nAnother line that is a little longer than the first one.",
            " ".join(f"Sentence number {i} of a very long paragraph." for i in range(80)),
            "x" * 450,  # one "sentence" longer than a part
        ]
    )
    parts = split_text(text, max_chars=400)
    assert all(len(p) <= 400 for p in parts) and len(parts) > 3
    assert "".join(text.split()) == "".join("".join(parts).split())
    assert split_text("short", max_chars=400) == ["short"] and split_text("  ") == []


def test_own_inputs_from_a_folder(tmp_path: Path) -> None:
    (tmp_path / "notice.txt").write_text("Legal notice to Ravi Kumar, Pune.", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "brief.md").write_text("# Brief\nClient: Asha Rao.", encoding="utf-8")
    (tmp_path / "case.json").write_text(
        json.dumps({"text": "Mr. Iyer was advised.", "title": "Case", "must_hide": ["Iyer"]}),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("how to use this folder", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    docs = {d.id: d for d in load_inputs(tmp_path)}
    assert set(docs) == {"notice", "brief", "case"}
    assert docs["case"].must_hide == ["Iyer"] and docs["case"].title == "Case"
    assert docs["brief"].text.startswith("# Brief")
    with pytest.raises(FileNotFoundError):
        load_inputs(tmp_path / "missing.txt")
    with pytest.raises(ValueError, match="unsupported"):
        real_docs.read_file(tmp_path / "image.png")


async def test_mock_run_saves_resumes_and_reports(tmp_path: Path) -> None:
    docs = [d for d in load_manifest(REAL) if d.id.startswith(("in08", "in10"))]
    docs.append(from_text("Our client Meera Nair of Kochi paid Rs 5,00,000.", id="pasted note"))
    said: list[str] = []
    results = await run_agents(docs, backend="mock", out_dir=tmp_path, progress=said.append)
    assert [r.ok for r in results] == [True, True, True]
    assert _saved(tmp_path) == {
        "in08_kar_hc_motor_accident.json",
        "in10_del_hc_procedural_order.json",
        "pasted_note.json",
    }
    long_doc = results[0]
    assert len(long_doc.parts) == 2 and all(len(p) <= 1800 for p in long_doc.parts)
    assert "Meera Nair" not in results[2].safe_text

    again = await run_agents(docs, backend="mock", out_dir=tmp_path, progress=said.append)
    assert sum("already done" in s for s in said) == 3
    assert [r.safe_text for r in again] == [r.safe_text for r in results]

    rows = summary_rows(results)
    assert rows[0]["document"] == "in08_kar_hc_motor_accident" and rows[0]["parts"] == 2
    assert totals(results)["documents failed"] == 0
    page = render_html(results[0])
    assert "<mark" in page and "Agent 2: facts" in page and "Agent 3" in page
    report = write_report(results, tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "must-hide identifiers hidden" in report and "## Yogesh Vig" in report


async def test_a_failing_part_is_recorded_and_the_run_goes_on(monkeypatch) -> None:
    async def broken(doc_id, text, llm):
        raise RuntimeError("model went away")

    monkeypatch.setattr("sanitise.mock_mode.mock_sanitise", broken)
    results = await run_agents([from_text("Ravi Kumar, Pune.")], backend="mock", progress=None)
    assert not results[0].ok and "model went away" in results[0].errors[0]
    assert summary_rows(results)[0]["status"].startswith("FAILED")
    assert totals(results)["documents failed"] == 1


def test_agent_settings_pick_the_backend() -> None:
    ollama = real_docs.agent_settings("ollama")
    assert (ollama.llm_backend, ollama.model, ollama.llm_num_ctx) == ("ollama", "gemma4:e4b", 8192)
    gemini = real_docs.agent_settings("gemini", "gemini-3.8-flash")
    assert gemini.llm_backend == "gemini" and gemini.model == gemini.gemini_model
    assert gemini.model == "gemini-3.8-flash"


def test_the_agents_notebook_runs_only_the_agents() -> None:
    notebook = json.loads(
        (REPO / "notebooks" / "legal_agents_kaggle.ipynb").read_text(encoding="utf-8")
    )
    code = "".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "-b law-pipeline-final" in code and "run_agents(" in code and "load_inputs(" in code
    for absent in ("monitor", "attack", "legal evaluate", "legal export", "finetune", "dashboard"):
        assert absent not in code.lower(), absent
