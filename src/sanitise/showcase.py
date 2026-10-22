"""Presentation views for the advisor demo.  OWNER: Person C (Platform and Integration).

Used by ``notebooks/demo_drive.ipynb`` and ``notebooks/scores_drive.ipynb``. Unlike
``sanitise.demo`` (the developer view), these views present Agent 1 as one system:
every span looks the same and carries a plain-language reason for its category.

    result = await detect(doc_id, text, llm)
    display(HTML(render_document(text, result)))
"""

from __future__ import annotations

import asyncio
import html
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sanitise.agents.detector.pipeline import detect
from sanitise.config import Settings, get_settings
from sanitise.demo import COLOURS
from sanitise.errors import ErrorCode, PipelineError
from sanitise.llm.base import LLMClient
from sanitise.llm.ollama import OllamaClient
from sanitise.loader import iter_documents
from sanitise.results import save_result
from sanitise.schemas import DetectorResult, PIICategory, SensitiveSpan, SpanMetrics

C = PIICategory

REASONS: dict[PIICategory, str] = {
    C.NAME: "Names a person directly.",
    C.AGE: "A person's age narrows down who they are.",
    C.DATE: "A specific date tied to an event in the person's life.",
    C.PHONE: "A contact number identifies the person directly.",
    C.EMAIL: "An email address identifies the person directly.",
    C.ID_NUMBER: "An ID number is linked to one person's records.",
    C.ADDRESS: "Shows where the person lives or can be found.",
    C.LOCATION: "A named place narrows down where the person is.",
    C.OCCUPATION: "A person's job; a rare job can single someone out.",
    C.RELATION: "Points to a specific family member.",
    C.PHYSICAL: "Body measurements help single the person out.",
    C.MEDICAL_CONDITION: "Health information about a specific person.",
    C.SYMPTOM: "A health complaint a specific person has; part of their medical picture.",
    C.QUASI_COMBO: "Details that identify the person when taken together.",
    C.OTHER: "Another detail that could identify the person.",
}

_FONT = "font-family:system-ui,-apple-system,Segoe UI,sans-serif"
_CARD = (
    f"background:#ffffff;color:#111827;{_FONT};border:1px solid #e5e7eb;border-radius:12px;"
    "padding:18px 20px;max-width:980px;line-height:1.5;font-size:14px"
)
_TH = "padding:6px 10px;text-align:left;color:#6b7280;font-weight:600"
_TD = "padding:6px 10px;border-top:1px solid #f3f4f6"


def _chip(category: PIICategory) -> str:
    return (
        f'<span style="background:{COLOURS[category]};border-radius:4px;padding:1px 7px;'
        f'font-size:11px;font-weight:600;white-space:nowrap">{category.value}</span>'
    )


def highlight(text: str, spans: list[SensitiveSpan]) -> str:
    """The document with every positional span marked the same way, coloured by category."""
    positional = sorted(
        (s for s in spans if s.start is not None and s.end is not None),
        key=lambda s: (s.start, -(s.end or 0)),
    )
    out: list[str] = []
    cursor = 0
    for span in positional:
        assert span.start is not None and span.end is not None
        if span.start < cursor:
            continue
        out.append(html.escape(text[cursor : span.start]))
        tip = f"{span.category.value}: {REASONS[span.category]}"
        out.append(
            f'<mark title="{html.escape(tip, quote=True)}" style="background:'
            f"{COLOURS[span.category]};color:#111827;border:1px solid rgba(0,0,0,.2);"
            f'border-radius:4px;padding:1px 4px">{html.escape(text[span.start : span.end])}'
            f'<sup style="font-size:9px;font-weight:700;margin-left:3px;opacity:.7">'
            f"{span.category.value}</sup></mark>"
        )
        cursor = span.end
    out.append(html.escape(text[cursor:]))
    return "".join(out).replace("\n", "<br>")


def render_document(text: str, result: DetectorResult) -> str:
    """Highlighted document, the combinations that identify the person, and a span table."""
    spans = result.output.spans
    positional = [s for s in spans if s.start is not None]
    combos = [s for s in spans if s.start is None]
    seconds = result.meta.total_ms / 1000

    if positional:
        rows = "".join(
            f'<tr><td style="{_TD}">{_chip(s.category)}</td>'
            f'<td style="{_TD}"><code>{html.escape(s.text)}</code></td>'
            f'<td style="{_TD};white-space:nowrap;color:#6b7280">{s.start}-{s.end}</td>'
            f'<td style="{_TD};color:#4b5563">{REASONS[s.category]}</td></tr>'
            for s in positional
        )
        table = (
            '<table style="border-collapse:collapse;width:100%;font-size:13px">'
            f'<tr><th style="{_TH}">category</th><th style="{_TH}">text</th>'
            f'<th style="{_TH}">position</th><th style="{_TH}">why it is sensitive</th></tr>'
            f"{rows}</table>"
        )
    else:
        table = '<p style="color:#6b7280;margin:4px 0">Nothing sensitive found.</p>'

    combo_html = (
        "".join(
            f'<div style="border-left:4px solid #65a30d;background:#f7fee7;padding:8px 12px;'
            f'margin:6px 0;border-radius:4px"><b>{html.escape(s.text)}</b><br>'
            f'<span style="color:#4b5563">{html.escape(s.rationale or "")}</span></div>'
            for s in combos
        )
        if combos
        else '<p style="color:#6b7280;margin:4px 0">None found.</p>'
    )
    legend = " ".join(_chip(c) for c in COLOURS if c is not C.QUASI_COMBO)
    return (
        f'<div style="{_CARD}">'
        '<div style="font-size:17px;font-weight:700">Agent 1 - Sensitive Information Detector'
        "</div>"
        f'<div style="color:#6b7280;font-size:12px;margin:2px 0 12px">doc_id {result.doc_id}'
        f" &nbsp;·&nbsp; {len(text)} characters</div>"
        f'<div style="margin-bottom:12px"><b>{len(spans)}</b> sensitive items found in '
        f"<b>{seconds:.2f} s</b> &nbsp;·&nbsp; every position checked against the text</div>"
        f'<div style="border:1px solid #e5e7eb;border-radius:8px;padding:14px;background:#f9fafb;'
        f'font-size:15px;line-height:2.1">{highlight(text, spans)}</div>'
        f'<div style="font-size:11px;color:#6b7280;margin:8px 0 16px">Hover over a highlight to '
        f"see why it is sensitive.<br>{legend}</div>"
        '<div style="font-weight:600;margin:4px 0">Combinations that identify the person together'
        f"</div>{combo_html}"
        f'<div style="font-weight:600;margin:16px 0 4px">All sensitive spans</div>{table}'
        "</div>"
    )


# ------------------------------------------------------------------ evaluation


async def run_fixtures(
    fixtures_dir: str | Path, out_dir: str | Path, llm: LLMClient
) -> list[DetectorResult]:
    """Run Agent 1 on every fixture and save one ``<doc_id>.json`` per document."""
    results = []
    for doc in iter_documents(fixtures_dir):
        result = await detect(doc.doc_id, doc.text, llm)
        save_result(result, out_dir)
        results.append(result)
    return results


def run_fixtures_sync(
    fixtures_dir: str | Path, out_dir: str | Path, settings: Settings | None = None
) -> list[DetectorResult]:
    """``run_fixtures`` with the configured Ollama model, callable without ``await``."""

    async def run() -> list[DetectorResult]:
        async with OllamaClient.from_settings(settings or get_settings()) as llm:
            return await run_fixtures(fixtures_dir, out_dir, llm)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, run()).result()


def render_error(exc: PipelineError) -> str:
    """A plain message for the viewer (the full envelope stays in the logs)."""
    if exc.code is ErrorCode.DOC_TOO_LONG:
        message = f"This document is too long ({exc.detail}). Please shorten it."
    elif exc.code in (ErrorCode.LLM_UNREACHABLE, ErrorCode.LLM_TIMEOUT):
        message = "The detector is not responding. Run cell 1 (Start) again, then retry."
    else:
        message = "This document could not be processed. Please try again."
    return (
        f'<div style="{_CARD};border-color:#fca5a5"><b style="color:#b91c1c">{exc.code.value}'
        f"</b><br>{html.escape(message)}</div>"
    )


def dataset_summary(fixtures_dir: str | Path, gold_dir: str | Path) -> dict[str, int]:
    fixtures = [
        json.loads(p.read_text(encoding="utf-8")) for p in Path(fixtures_dir).glob("*.json")
    ]
    gold = [json.loads(p.read_text(encoding="utf-8")) for p in Path(gold_dir).glob("*.json")]
    return {
        "documents": len(gold),
        "patient_questions": sum("chatdoctor" in f.get("source", "") for f in fixtures),
        "emails": sum("enron" in f.get("source", "") for f in fixtures),
        "gold_spans": sum(len(g["spans"]) for g in gold),
        "combinations": sum(s["start"] is None for g in gold for s in g["spans"]),
        "nothing_sensitive": sum(not g["spans"] for g in gold),
    }


def _big(label: str, value: float, primary: bool = False) -> str:
    colour = "#065f46" if primary else "#111827"
    note = '<div style="font-size:11px;color:#6b7280">primary figure</div>' if primary else ""
    return (
        f'<div style="flex:1;min-width:150px;border:1px solid #e5e7eb;border-radius:10px;'
        f'padding:12px 16px;background:{"#ecfdf5" if primary else "#f9fafb"}">'
        f'<div style="color:#6b7280;font-size:12px;font-weight:600">{label}</div>'
        f'<div style="font-size:34px;font-weight:700;color:{colour}">{value:.2f}</div>{note}</div>'
    )


def render_scores(
    metrics: SpanMetrics,
    dataset: dict[str, int],
    system: str = "rules + llama3:8b",
    note: str | None = None,
) -> str:
    """Headline precision / recall / F1, the dataset, and the per-category table.

    ``note`` is shown under the title, e.g. which run and gold-set version is scored.
    """
    cats = metrics.per_category
    tp = int(sum(v.get("tp", 0) for v in cats.values()))
    fp = int(sum(v.get("fp", 0) for v in cats.values()))
    fn = int(sum(v.get("fn", 0) for v in cats.values()))
    rows = "".join(
        f'<tr><td style="{_TD}">{_chip(C(cat))}</td>'
        f'<td style="{_TD};text-align:right">{int(v["support"])}</td>'
        f'<td style="{_TD};text-align:right"><b>{v["recall"]:.2f}</b></td>'
        f'<td style="{_TD};text-align:right">{v["precision"]:.2f}</td>'
        f'<td style="{_TD};text-align:right">{v["f1"]:.2f}</td></tr>'
        for cat, v in sorted(cats.items(), key=lambda kv: -kv[1]["support"])
    )
    d = dataset
    return (
        f'<div style="{_CARD}">'
        '<div style="font-size:17px;font-weight:700">Agent 1 - measured accuracy</div>'
        f'<div style="color:#6b7280;font-size:12px;margin:2px 0 14px">System: Agent 1 '
        f"({html.escape(system)}) &nbsp;·&nbsp; scored against hand-labelled gold annotations"
        "</div>"
        + (
            f'<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:6px;'
            f'padding:6px 10px;font-size:12px;margin-bottom:12px">{html.escape(note)}</div>'
            if note
            else ""
        )
        + ""
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px">'
        f"{_big('Recall', metrics.recall, primary=True)}{_big('Precision', metrics.precision)}"
        f"{_big('F1', metrics.f1)}</div>"
        f'<div style="margin-bottom:14px"><b>{tp}</b> of <b>{tp + fn}</b> gold items found '
        f"&nbsp;·&nbsp; <b>{fn}</b> missed &nbsp;·&nbsp; <b>{fp}</b> extra flags</div>"
        '<div style="font-weight:600;margin:4px 0">Dataset</div>'
        f'<div style="color:#374151;margin-bottom:14px">{d["documents"]} documents '
        f"({d['patient_questions']} patient questions to a doctor, {d['emails']} workplace "
        f"emails), {d['gold_spans']} hand-labelled sensitive items including "
        f"{d['combinations']} identifying combinations; {d['nothing_sensitive']} documents "
        "contain nothing sensitive, to measure false alarms. Each document was annotated by "
        "one person and reviewed by a second.</div>"
        '<div style="font-weight:600;margin:4px 0">Per category</div>'
        '<table style="border-collapse:collapse;font-size:13px;min-width:420px">'
        f'<tr><th style="{_TH}">category</th><th style="{_TH};text-align:right">gold items</th>'
        f'<th style="{_TH};text-align:right">recall</th>'
        f'<th style="{_TH};text-align:right">precision</th>'
        f'<th style="{_TH};text-align:right">F1</th></tr>{rows}</table>'
        '<div style="font-size:12px;color:#6b7280;margin-top:12px">Match rule: a predicted '
        "item counts when it overlaps a gold item of the same category; combinations are "
        "matched on category; each gold item can be credited once. Recall is the primary "
        "figure because a missed item is a privacy leak.</div>"
        "</div>"
    )
