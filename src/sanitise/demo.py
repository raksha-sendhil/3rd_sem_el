"""Notebook / demo helpers.  OWNER: Person C (Platform and Integration).

``analyse()`` runs the full Agent 1 pipeline on one piece of text in-process, and
``to_html()`` renders the result as a highlighted document, a span table and the
quasi-identifier combinations. Used by ``notebooks/colab_live_test.ipynb``::

    result, trace = await analyse(text, model="llama3:8b")
    display(HTML(to_html(text, result, trace)))
"""

from __future__ import annotations

import asyncio
import html
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sanitise.agents.detector.pipeline import DetectorTrace, detect_with_trace
from sanitise.config import Settings, get_settings
from sanitise.errors import PipelineError
from sanitise.evaluation import scoring
from sanitise.llm.base import LLMClient
from sanitise.llm.mock import MockLLM
from sanitise.llm.ollama import OllamaClient
from sanitise.loader import compute_doc_id
from sanitise.results import load_gold, load_predictions
from sanitise.schemas import DetectorResult, PIICategory, SensitiveSpan

C = PIICategory

# Light backgrounds with dark text: readable in Colab's light and dark themes alike,
# because the whole block sits on its own white card.
COLOURS: dict[PIICategory, str] = {
    C.NAME: "#fde68a",
    C.AGE: "#fecaca",
    C.DATE: "#bfdbfe",
    C.PHONE: "#c7d2fe",
    C.EMAIL: "#ddd6fe",
    C.ID_NUMBER: "#fbcfe8",
    C.ADDRESS: "#fed7aa",
    C.LOCATION: "#bbf7d0",
    C.OCCUPATION: "#a5f3fc",
    C.RELATION: "#e9d5ff",
    C.PHYSICAL: "#fef3c7",
    C.MEDICAL_CONDITION: "#fca5a5",
    C.SYMPTOM: "#f5d0fe",
    C.QUASI_COMBO: "#d9f99d",
    C.OTHER: "#e5e7eb",
}

_CARD = (
    "background:#ffffff;color:#111827;font-family:system-ui,-apple-system,Segoe UI,sans-serif;"
    "border:1px solid #e5e7eb;border-radius:10px;padding:16px 18px;max-width:980px;"
    "line-height:1.5;font-size:14px"
)


async def analyse(
    text: str,
    llm: LLMClient | None = None,
    *,
    model: str | None = None,
    rules_only: bool = False,
    settings: Settings | None = None,
) -> tuple[DetectorResult, DetectorTrace]:
    """Run Agent 1 on ``text``. Builds (and closes) an Ollama client unless ``llm`` is given."""
    settings = settings or get_settings()
    doc_id = compute_doc_id(text)
    if rules_only:
        llm = llm or MockLLM(model="rules-only")
    if llm is not None:
        return await detect_with_trace(
            doc_id, text, llm, run_llm=not rules_only, max_chars=settings.max_doc_chars
        )
    async with OllamaClient.from_settings(settings, model=model) as client:
        return await detect_with_trace(doc_id, text, client, max_chars=settings.max_doc_chars)


def analyse_sync(text: str, **kwargs: Any) -> tuple[DetectorResult, DetectorTrace]:
    """``analyse()`` for code that cannot ``await`` (widget callbacks, plain scripts).

    Runs on a worker thread with its own event loop, so it works the same inside a
    notebook whose loop is already running.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, analyse(text, **kwargs)).result()


# ------------------------------------------------------------------ HTML


def _chip(category: PIICategory) -> str:
    return (
        f'<span style="background:{COLOURS[category]};border-radius:4px;padding:1px 6px;'
        f'font-size:11px;font-weight:600;letter-spacing:.02em">{category.value}</span>'
    )


def _found_by(span: SensitiveSpan) -> str:
    return "rules" if span.method == "rule" else "LLM"


def highlight(text: str, spans: list[SensitiveSpan]) -> str:
    """The document with every positional span wrapped in a coloured, labelled mark."""
    positional = sorted(
        (s for s in spans if s.start is not None and s.end is not None),
        key=lambda s: (s.start, -(s.end or 0)),
    )
    out: list[str] = []
    cursor = 0
    for span in positional:
        assert span.start is not None and span.end is not None
        if span.start < cursor:  # merged output never overlaps; be safe anyway
            continue
        out.append(html.escape(text[cursor : span.start]))
        border = "1px solid rgba(0,0,0,.25)" if span.method == "rule" else "1px dashed #111827"
        tip = f"{span.category.value} - found by {_found_by(span)}"
        if span.rationale:
            tip += f" - {span.rationale}"
        out.append(
            f'<mark title="{html.escape(tip, quote=True)}" style="background:'
            f"{COLOURS[span.category]};color:#111827;border:{border};border-radius:4px;"
            f'padding:1px 3px">{html.escape(text[span.start : span.end])}'
            f'<sup style="font-size:9px;font-weight:700;margin-left:3px;opacity:.75">'
            f"{span.category.value}</sup></mark>"
        )
        cursor = span.end
    out.append(html.escape(text[cursor:]))
    return "".join(out).replace("\n", "<br>")


def to_html(
    text: str,
    result: DetectorResult,
    trace: DetectorTrace | None = None,
    *,
    title: str = "Agent 1 - Sensitive Information Detector",
) -> str:
    spans = result.output.spans
    positional = [s for s in spans if s.start is not None]
    combos = [s for s in spans if s.start is None]
    meta = result.meta
    rule_n = sum(s.method == "rule" for s in spans)
    llm_n = len(spans) - rule_n
    llm_ran = trace.llm_ran if trace is not None else meta.llm_ms > 0

    summary = (
        f"<b>{len(spans)}</b> sensitive items &nbsp;·&nbsp; <b>{rule_n}</b> from rules "
        f"(Presidio + spaCy + regex, {meta.rule_ms / 1000:.2f} s) &nbsp;·&nbsp; "
    )
    if llm_ran:
        summary += (
            f"<b>{llm_n}</b> from the LLM ({html.escape(meta.model)}, constrained JSON, "
            f"{meta.llm_ms / 1000:.2f} s, retries {meta.retries})"
        )
    else:
        summary += "LLM pass skipped (rules only)"
    summary += f" &nbsp;·&nbsp; total {meta.total_ms / 1000:.2f} s &nbsp;·&nbsp; validation OK"

    rows = "".join(
        "<tr>"
        f'<td style="padding:4px 8px">{_chip(s.category)}</td>'
        f'<td style="padding:4px 8px"><code>{html.escape(s.text)}</code></td>'
        f'<td style="padding:4px 8px;white-space:nowrap">{s.start}-{s.end}</td>'
        f'<td style="padding:4px 8px">{_found_by(s)}</td>'
        f'<td style="padding:4px 8px;color:#4b5563">{html.escape(s.rationale or "")}</td>'
        "</tr>"
        for s in positional
    )
    table = (
        '<table style="border-collapse:collapse;width:100%;font-size:13px">'
        '<tr style="text-align:left;color:#6b7280;border-bottom:1px solid #e5e7eb">'
        '<th style="padding:4px 8px">category</th><th style="padding:4px 8px">text</th>'
        '<th style="padding:4px 8px">chars</th><th style="padding:4px 8px">found by</th>'
        f'<th style="padding:4px 8px">why</th></tr>{rows}</table>'
        if positional
        else '<p style="color:#6b7280">No positional spans.</p>'
    )
    combo_html = (
        "".join(
            f'<div style="border-left:4px solid #65a30d;background:#f7fee7;padding:6px 10px;'
            f'margin:6px 0;border-radius:4px"><b>{html.escape(s.text)}</b><br>'
            f'<span style="color:#4b5563">{html.escape(s.rationale or "")}</span></div>'
            for s in combos
        )
        if combos
        else '<p style="color:#6b7280">None found.</p>'
    )
    legend = " ".join(_chip(c) for c in COLOURS if c is not C.QUASI_COMBO)
    return (
        f'<div style="{_CARD}">'
        f'<div style="font-size:16px;font-weight:700;margin-bottom:2px">{html.escape(title)}</div>'
        f'<div style="color:#6b7280;font-size:12px;margin-bottom:10px">doc_id {result.doc_id}'
        f" &nbsp;·&nbsp; {len(text)} chars</div>"
        f'<div style="margin-bottom:12px">{summary}</div>'
        f'<div style="border:1px solid #e5e7eb;border-radius:8px;padding:12px;background:#f9fafb;'
        f'font-size:15px;line-height:2">{highlight(text, spans)}</div>'
        f'<div style="font-size:11px;color:#6b7280;margin:6px 0 14px">Solid border: rules. '
        f"Dashed border: LLM (quote grounded to exact offsets). Hover a span for the reason."
        f"<br>{legend}</div>"
        f'<div style="font-weight:600;margin:6px 0">Quasi-identifier combinations (LLM)</div>'
        f"{combo_html}"
        f'<div style="font-weight:600;margin:14px 0 6px">All positional spans</div>{table}'
        "</div>"
    )


# ------------------------------------------------------------------ model comparison


def summarise_run(label: str, pred_dir: str | Path, gold_dir: str | Path) -> dict[str, Any]:
    """Scores plus speed for one ``detect --out`` directory (one row of the comparison)."""
    gold = load_gold(gold_dir)
    pred = load_predictions(pred_dir)
    metrics = scoring.score(pred, gold)
    metas = [
        DetectorResult.model_validate_json(p.read_text(encoding="utf-8")).meta
        for p in sorted(Path(pred_dir).glob("*.json"))
    ]
    per_cat = metrics.per_category
    return {
        "run": label,
        "documents": len(pred),
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "quasi_recall": per_cat.get("QUASI_COMBO", {}).get("recall", 0.0),
        "medical_recall": per_cat.get("MEDICAL_CONDITION", {}).get("recall", 0.0),
        "llm_s_per_doc": sum(m.llm_ms for m in metas) / max(len(metas), 1) / 1000,
        "retries": sum(m.retries for m in metas),
    }


def comparison_markdown(rows: list[dict[str, Any]], gold_spans: int) -> str:
    head = (
        "| run | docs | recall | precision | F1 | QUASI_COMBO recall | "
        "MEDICAL_CONDITION recall | LLM s/doc | retries |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        f"| {r['run']} | {r['documents']} | **{r['recall']:.2f}** | {r['precision']:.2f} | "
        f"{r['f1']:.2f} | {r['quasi_recall']:.2f} | {r['medical_recall']:.2f} | "
        f"{r['llm_s_per_doc']:.1f} | {r['retries']} |\n"
        for r in rows
    )
    return (
        "# Agent 1 - model comparison\n\n"
        f"20 fixtures, {gold_spans} gold spans, match rule from spec 3.2 (one-to-one). "
        "Recall is the primary figure.\n\n" + head + body
    )


def error_html(exc: PipelineError) -> str:
    """The spec 8.5 error envelope, shown as a card."""
    body = html.escape(exc.to_envelope().model_dump_json(indent=2))
    return (
        f'<div style="{_CARD};border-color:#fca5a5"><b style="color:#b91c1c">'
        f"{exc.code.value}</b> at stage <code>{exc.stage}</code>"
        f'<pre style="background:#fef2f2;padding:8px;border-radius:6px;'
        f'white-space:pre-wrap">{body}</pre></div>'
    )
