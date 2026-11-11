"""Agents 1-3 on real legal documents and on your own files, with readable results.

OWNER: Person C. This is the agents-only path used by ``notebooks/legal_agents_kaggle.ipynb``:
no store, no search, no answers, no attacks and no session monitor. A document goes through
Agent 1 (find), Agent 2 (facts) and Agent 3 (rewrite + leak check), and the result is shown
and saved.

    docs = load_manifest()                       # the 14 public documents in data/legal/real/
    docs += load_inputs("my_inputs/")            # .txt .md .json .pdf .docx, files or folders
    docs.append(from_text("...pasted text..."))
    results = await run_agents(docs, backend="ollama", out_dir="agent_runs")
    summary_rows(results); render_html(results[0]); write_report(results, "agent_runs/REPORT.md")

Long documents are split into parts of at most ``max_chars`` characters (1,800 by default) at
paragraph breaks: the agents were built for notes of a few hundred characters, a whole
judgment does not fit the model's context, and Agent 1 refuses anything over
``max_doc_chars`` (2,000). Each part goes through the agents on its own, so a person in two
parts may get a different placeholder in each.
"""

from __future__ import annotations

import html
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sanitise.schemas import PIICategory, SanitisedRecord

_IN_REPO = Path(__file__).resolve().parents[3] / "data" / "legal" / "real"
REAL_DIR = _IN_REPO if _IN_REPO.is_dir() else Path("data/legal/real")
SUFFIXES = (".txt", ".md", ".json", ".pdf", ".docx")
# Agent 1 categories that name a person or a record outright; any of these still in the safe
# text is reported as left behind (the same set the leak check treats as direct).
DIRECT = frozenset(
    {
        PIICategory.NAME,
        PIICategory.PHONE,
        PIICategory.EMAIL,
        PIICategory.ID_NUMBER,
        PIICategory.CASE_NUMBER,
        PIICategory.FINANCIAL,
        PIICategory.ADDRESS,
    }
)


@dataclass
class Document:
    id: str
    text: str
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    must_hide: list[str] = field(default_factory=list)  # identifiers that must not survive
    must_keep: list[str] = field(default_factory=list)  # legal substance that must survive


# ---------------------------------------------------------------- loading


def load_manifest(root: str | Path = REAL_DIR) -> list[Document]:
    """The public documents listed in ``<root>/manifest.json``."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    docs = []
    for entry in manifest["documents"]:
        text = (root / entry["file"]).read_text(encoding="utf-8")
        meta = {
            k: v for k, v in entry.items() if k not in ("id", "title", "must_hide", "must_keep")
        }
        docs.append(
            Document(
                id=entry["id"],
                text=text,
                title=entry.get("title", ""),
                meta=meta,
                must_hide=list(entry.get("must_hide", [])),
                must_keep=list(entry.get("must_keep", [])),
            )
        )
    return docs


SETS = {
    "indian300": REAL_DIR / "indian300",  # 300 Indian judgments, checks from court metadata
    "curated": REAL_DIR,  # 14 hand-picked documents, checks written by hand
}


def load_set(name: str = "indian300", limit: int | None = None) -> list[Document]:
    """A document set by name: "indian300", "curated" or "both" (curated first). ``limit``
    keeps the first N, for a quick trial."""
    names = ("curated", "indian300") if name == "both" else (name,)
    unknown = [n for n in names if n not in SETS]
    if unknown:
        raise ValueError(f"unknown set {unknown[0]!r}: use one of {[*SETS, 'both']}")
    docs = [d for n in names for d in load_manifest(SETS[n])]
    return docs[:limit] if limit else docs


def read_file(path: str | Path) -> str:
    """The text of one .txt, .md, .json, .pdf or .docx file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "text" in data:
            return str(data["text"])
        raise ValueError(f"{path}: a .json input needs a 'text' field")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("PDF inputs need pypdf: pip install pypdf") from exc
        pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
        text = "\n".join(pages)
        if not text.strip():
            raise ValueError(f"{path}: no text layer (a scanned PDF needs OCR first)")
        return text
    if suffix == ".docx":
        try:
            import docx
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("Word inputs need python-docx: pip install python-docx") from exc
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    raise ValueError(f"{path}: unsupported file type (use {', '.join(SUFFIXES)})")


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "doc"


def load_inputs(*paths: str | Path) -> list[Document]:
    """Your own documents: files, or folders searched recursively for supported files.

    A ``.json`` file may also give ``title``, ``must_hide`` and ``must_keep`` (lists of
    strings) to get the same spot checks as the public documents."""
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUFFIXES)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    docs = []
    for f in files:
        if f.name.lower() == "readme.md":
            continue
        extra: dict[str, Any] = {}
        if f.suffix.lower() == ".json":
            extra = json.loads(f.read_text(encoding="utf-8"))
        docs.append(
            Document(
                id=_slug(f.stem),
                text=read_file(f),
                title=str(extra.get("title", f.name)),
                meta={"file": str(f), "source": "your input"},
                must_hide=list(extra.get("must_hide", [])),
                must_keep=list(extra.get("must_keep", [])),
            )
        )
    return docs


def from_text(
    text: str,
    id: str = "pasted",
    title: str = "",
    must_hide: Iterable[str] = (),
    must_keep: Iterable[str] = (),
) -> Document:
    """A document typed or pasted into the notebook."""
    return Document(
        id=_slug(id),
        text=text,
        title=title or id,
        meta={"source": "pasted"},
        must_hide=list(must_hide),
        must_keep=list(must_keep),
    )


# ---------------------------------------------------------------- splitting

_SENTENCE_END = re.compile(r"(?<=[.;:?!])\s+")


def split_text(text: str, max_chars: int = 1800) -> list[str]:
    """Parts of at most about ``max_chars`` characters, cut at blank lines, then line breaks,
    then sentence ends. Text shorter than ``max_chars`` comes back whole."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        for line in para.split("\n"):
            if len(line) <= max_chars:
                pieces.append(line)
                continue
            sentence = ""
            for s in _SENTENCE_END.split(line):
                if sentence and len(sentence) + 1 + len(s) > max_chars:
                    pieces.append(sentence)
                    sentence = s
                else:
                    sentence = f"{sentence} {s}" if sentence else s
            while len(sentence) > max_chars:  # one sentence longer than a part
                pieces.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if sentence:
                pieces.append(sentence)
    parts: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > max_chars:
            parts.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        parts.append(current)
    return parts


# ---------------------------------------------------------------- running


@dataclass
class DocResult:
    doc: Document
    parts: list[str]
    records: list[SanitisedRecord | None]
    errors: list[str]
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.records) and all(r is not None for r in self.records)

    def _done(self) -> list[SanitisedRecord]:
        return [r for r in self.records if r is not None]

    @property
    def safe_text(self) -> str:
        return "\n\n".join(r.rewriter.safe_text for r in self._done())

    def spans(self) -> list:
        return [s for r in self._done() for s in r.detector.spans]

    def facts(self) -> list:
        return [f for r in self._done() for f in r.extractor.facts]

    def decisions(self) -> list:
        return [d for r in self._done() for d in r.rewriter.decisions]

    def fallbacks(self) -> list[str]:
        return [f for r in self._done() for f in r.meta.fallback]

    def checks(self) -> dict[str, list[str]]:
        """Spot checks on the safe text: which must_hide strings leaked, which must_keep
        strings were dropped, and which direct identifiers Agent 1 found are still there."""
        safe = self.safe_text.lower()
        found_direct = {s.text for s in self.spans() if s.category in DIRECT and len(s.text) > 2}
        return {
            "leaked": [s for s in self.doc.must_hide if s.lower() in safe],
            "hidden": [s for s in self.doc.must_hide if s.lower() not in safe],
            "dropped": [s for s in self.doc.must_keep if s.lower() not in safe],
            "kept": [s for s in self.doc.must_keep if s.lower() in safe],
            "direct_left": sorted(t for t in found_direct if t.lower() in safe),
            "residual_leaks": [s.text for r in self._done() for s in r.residual_leaks],
        }


def agent_settings(
    backend: str = "ollama",
    model: str | None = None,
    *,
    num_ctx: int = 8192,
    num_predict: int = 3072,
    timeout_s: float = 300.0,
    **overrides: Any,
):
    """The legal profile's settings with the model and limits the notebook chooses.

    backend "ollama" (Gemma 4 on the notebook's own GPU; text never leaves it), "gemini"
    (Google's API: the original text goes to Google) or "mock" (rules only, no model)."""
    from sanitise.config import get_settings
    from sanitise.domain import settings_for

    base = get_settings().model_copy(update={"llm_backend": backend})
    settings = settings_for(base, "legal")
    update: dict[str, Any] = {
        "llm_num_ctx": num_ctx,
        "llm_num_predict": num_predict,
        "llm_timeout_s": timeout_s,
        **overrides,
    }
    if model:
        update["gemini_model" if backend == "gemini" else "model"] = model
        if backend == "gemini":
            update["model"] = model
    return settings.model_copy(update=update)


def _record_path(out_dir: Path, doc: Document) -> Path:
    return out_dir / f"{doc.id}.json"


def _load_saved(path: Path, doc: Document, parts: list[str]) -> DocResult | None:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if saved.get("parts") != parts or not all(saved.get("records", [None])):
        return None  # the text or the split changed, or a part failed: run again
    records = [SanitisedRecord.model_validate(r) for r in saved["records"]]
    return DocResult(doc, parts, records, [], saved.get("seconds", 0.0))


def save_result(result: DocResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = _record_path(out, result.doc)
    payload = {
        "id": result.doc.id,
        "title": result.doc.title,
        "meta": result.doc.meta,
        "parts": result.parts,
        "safe_text": result.safe_text,
        "checks": result.checks(),
        "errors": result.errors,
        "seconds": result.seconds,
        "records": [r.model_dump(mode="json") if r else None for r in result.records],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


async def run_agents(
    docs: Iterable[Document],
    *,
    backend: str = "ollama",
    model: str | None = None,
    max_chars: int = 1800,
    out_dir: str | Path | None = None,
    rerun: bool = False,
    progress: Callable[[str], None] | None = print,
    settings: Any = None,
) -> list[DocResult]:
    """Every document through Agents 1, 2 and 3, part by part.

    With ``out_dir`` each document is saved as ``<out_dir>/<id>.json`` and a later call skips
    documents already done with the same text (``rerun=True`` runs them again). A part whose
    model call fails is recorded in ``errors`` and the run goes on to the next document."""
    from sanitise.agents.pipeline import sanitise
    from sanitise.config import get_settings
    from sanitise.domain import use_domain
    from sanitise.llm.factory import build_agent1_llm, build_llm_client, close_llm_client

    say = progress or (lambda _msg: None)
    max_chars = min(max_chars, get_settings().max_doc_chars)  # Agent 1 refuses longer text
    out = Path(out_dir) if out_dir else None
    mock = backend == "mock"
    if mock:
        from sanitise.mock_mode import mock_llm, mock_sanitise

        llm = detector = mock_llm()
    else:
        settings = settings or agent_settings(backend, model)
        llm = build_llm_client(settings)
        detector = build_agent1_llm(settings, llm)
    results: list[DocResult] = []
    docs = list(docs)
    try:
        with use_domain("legal"):
            for n, doc in enumerate(docs, 1):
                parts = split_text(doc.text, max_chars)
                if out and not rerun:
                    saved = _load_saved(_record_path(out, doc), doc, parts)
                    if saved:
                        say(f"[{n}/{len(docs)}] {doc.id}: already done, loaded from {out}")
                        results.append(saved)
                        continue
                records: list[SanitisedRecord | None] = []
                errors: list[str] = []
                t0 = time.perf_counter()
                for i, part in enumerate(parts, 1):
                    part_id = doc.id if len(parts) == 1 else f"{doc.id}#p{i}"
                    try:
                        if mock:
                            records.append(await mock_sanitise(part_id, part, llm))
                        else:
                            records.append(
                                await sanitise(part_id, part, llm, detector_llm=detector)
                            )
                    except Exception as exc:  # one bad part must not stop the run
                        records.append(None)
                        errors.append(f"part {i}: {type(exc).__name__}: {exc}")
                result = DocResult(doc, parts, records, errors, time.perf_counter() - t0)
                results.append(result)
                if out:
                    save_result(result, out)
                c = result.checks()
                status = "ok" if result.ok else f"{len(errors)} part(s) failed"
                say(
                    f"[{n}/{len(docs)}] {doc.id}: {status}; {len(parts)} part(s), "
                    f"{len(result.spans())} spans, {len(result.facts())} facts, "
                    f"{len(c['leaked'])} must-hide leaked, {result.seconds:.0f} s"
                    + (f"; fallbacks {Counter(result.fallbacks())}" if result.fallbacks() else "")
                )
    finally:
        if not mock:
            if detector is not llm:
                await close_llm_client(detector)
            await close_llm_client(llm)
    return results


# ---------------------------------------------------------------- reporting


def summary_rows(results: Iterable[DocResult]) -> list[dict[str, Any]]:
    """One row per document, for a table (``pandas.DataFrame(summary_rows(results))``)."""
    rows = []
    for r in results:
        c = r.checks()
        cats = Counter(s.category.value for s in r.spans())
        rows.append(
            {
                "document": r.doc.id,
                "source": r.doc.meta.get("court") or r.doc.meta.get("source", ""),
                "chars": len(r.doc.text),
                "parts": len(r.parts),
                "status": "ok" if r.ok else "FAILED " + "; ".join(r.errors)[:80],
                "spans": len(r.spans()),
                "top categories": ", ".join(f"{k} {v}" for k, v in cats.most_common(4)),
                "facts": len(r.facts()),
                "must-hide hidden": f"{len(c['hidden'])}/{len(r.doc.must_hide)}"
                if r.doc.must_hide
                else "-",
                "leaked": ", ".join(c["leaked"]),
                "must-keep kept": f"{len(c['kept'])}/{len(r.doc.must_keep)}"
                if r.doc.must_keep
                else "-",
                "dropped": ", ".join(c["dropped"]),
                "direct ids left": len(c["direct_left"]),
                "fallbacks": ", ".join(sorted(set(r.fallbacks()))),
                "seconds": round(r.seconds, 1),
            }
        )
    return rows


def totals(results: Iterable[DocResult]) -> dict[str, Any]:
    """The run in a few numbers."""
    results = list(results)
    must_hide = sum(len(r.doc.must_hide) for r in results if r.ok)
    leaked = sum(len(r.checks()["leaked"]) for r in results if r.ok)
    must_keep = sum(len(r.doc.must_keep) for r in results if r.ok)
    kept = sum(len(r.checks()["kept"]) for r in results if r.ok)
    falls = Counter(f for r in results for f in r.fallbacks())
    parts = sum(len(r.records) for r in results)
    return {
        "documents": len(results),
        "documents failed": sum(not r.ok for r in results),
        "parts": parts,
        "must-hide identifiers hidden": f"{must_hide - leaked}/{must_hide}"
        + (f" ({(must_hide - leaked) / must_hide:.0%})" if must_hide else ""),
        "must-keep terms kept": f"{kept}/{must_keep}"
        + (f" ({kept / must_keep:.0%})" if must_keep else ""),
        "direct identifiers left (Agent 1 found, Agent 3 kept)": sum(
            len(r.checks()["direct_left"]) for r in results
        ),
        "parts with a fallback": dict(falls),
        "median seconds per document": sorted(r.seconds for r in results)[len(results) // 2]
        if results
        else 0,
    }


_COLOURS = {
    "NAME": "#ffd6d6",
    "ADDRESS": "#ffe4c2",
    "LOCATION": "#fff1b8",
    "PHONE": "#e3d4ff",
    "EMAIL": "#e3d4ff",
    "ID_NUMBER": "#d6e4ff",
    "CASE_NUMBER": "#d6e4ff",
    "FINANCIAL": "#d6e4ff",
    "DATE": "#d9f2e3",
    "AGE": "#d9f2e3",
    "MONEY": "#e8f5c8",
    "ORGANIZATION": "#f5d9f0",
    "COURT": "#ececec",
    "OCCUPATION": "#d4f1f4",
    "RELATION": "#fde2f3",
    "MEDICAL_CONDITION": "#ffccd5",
    "SYMPTOM": "#ffccd5",
}


def _highlight(text: str, spans: list) -> str:
    located = sorted(
        (s for s in spans if s.start is not None and s.end is not None), key=lambda s: s.start
    )
    out, pos = [], 0
    for s in located:
        if s.start < pos:
            continue  # overlapping span: the first one wins
        out.append(html.escape(text[pos : s.start]))
        colour = _COLOURS.get(s.category.value, "#eeeeee")
        out.append(
            f'<mark style="background:{colour};color:#111;padding:0 2px;border-radius:3px" '
            f'title="{s.category.value} ({s.method})">{html.escape(text[s.start : s.end])}'
            f'<sub style="font-size:9px;color:#444">{s.category.value}</sub></mark>'
        )
        pos = s.end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "<p><i>none</i></p>"
    head = "".join(f"<th style='text-align:left;padding:2px 8px'>{h}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td style='padding:2px 8px'>{html.escape(str(v))}</td>" for v in row)
        + "</tr>"
        for row in rows
    )
    return f"<table style='border-collapse:collapse;font-size:12px'><tr>{head}</tr>{body}</table>"


def render_html(result: DocResult) -> str:
    """One document: Agent 1's spans on the original, Agent 2's facts, Agent 3's routing and
    the safe text, and the spot checks. ``IPython.display.HTML(render_html(r))``."""
    doc = result.doc
    box = "white-space:pre-wrap;font-family:Georgia,serif;font-size:13px;line-height:1.5;"
    box += "border:1px solid #ccc;padding:8px;color:#111;background:#fff"
    original = []
    for i, (part, rec) in enumerate(zip(result.parts, result.records, strict=True), 1):
        label = f"<b>part {i}</b>\n" if len(result.parts) > 1 else ""
        original.append(
            label + (_highlight(part, rec.detector.spans) if rec else html.escape(part))
        )
    c = result.checks()
    meta = doc.meta
    source = ""
    if meta.get("url"):
        source = (
            f"<a href='{html.escape(meta['url'])}' target='_blank'>"
            f"{html.escape(meta.get('source', 'source'))}</a>"
            f" &middot; {html.escape(meta.get('licence', ''))}"
        )
    parts = [
        f"<h3 style='margin-bottom:2px'>{html.escape(doc.title or doc.id)}</h3>",
        f"<div style='font-size:12px;color:#555'>{html.escape(doc.id)} &middot; "
        f"{html.escape(str(meta.get('doc_type', meta.get('source', ''))))} &middot; {source}</div>",
    ]
    if result.errors:
        parts.append(f"<p style='color:#b00'>Failed: {html.escape('; '.join(result.errors))}</p>")
    parts += [
        "<div style='display:flex;gap:12px;flex-wrap:wrap'>",
        f"<div style='flex:1;min-width:320px'><h4>Original, with Agent 1's spans</h4>"
        f"<div style='{box}'>{''.join(original)}</div></div>",
        f"<div style='flex:1;min-width:320px'><h4>Agent 3's safe text</h4>"
        f"<div style='{box}'>{html.escape(result.safe_text)}</div></div>",
        "</div>",
        "<h4>Spot checks</h4>",
        _table(
            ["check", "result"],
            [
                [
                    "must-hide still in safe text (should be empty)",
                    ", ".join(c["leaked"]) or "none",
                ],
                ["must-keep dropped (should be empty)", ", ".join(c["dropped"]) or "none"],
                [
                    "direct identifiers Agent 1 found that are still there",
                    ", ".join(c["direct_left"]) or "none",
                ],
                ["leak check residuals", ", ".join(c["residual_leaks"]) or "none"],
                ["fallbacks", ", ".join(result.fallbacks()) or "none"],
            ],
        ),
        "<h4>Agent 2: facts</h4>",
        _table(
            ["subject", "relation", "importance", "value"],
            [[f.subject, f.relation, f.importance.value, f.value] for f in result.facts()],
        ),
        "<h4>Agent 3: what happened to each span</h4>",
        _table(
            ["category", "action", "original", "replacement"],
            [
                [d.category.value, d.action.value, d.span_text, d.replacement or ""]
                for d in result.decisions()
            ],
        ),
    ]
    return "\n".join(parts)


def write_report(results: Iterable[DocResult], path: str | Path) -> Path:
    """REPORT.md: the totals, the per-document table, and each document's safe text."""
    results = list(results)
    lines = ["# Agents 1-3 on real legal documents", ""]
    for k, v in totals(results).items():
        lines.append(f"- **{k}**: {v}")
    rows = summary_rows(results)
    if rows:
        cols = list(rows[0])
        lines += ["", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for row in rows:
            cells = [str(row[c]).replace("|", "/").replace("\n", " ") for c in cols]
            lines.append("| " + " | ".join(cells) + " |")
    for r in results:
        c = r.checks()
        lines += [
            "",
            f"## {r.doc.title or r.doc.id}",
            "",
            f"`{r.doc.id}` - {r.doc.meta.get('url', r.doc.meta.get('file', ''))}",
            "",
            f"- leaked: {', '.join(c['leaked']) or 'none'}; dropped: "
            f"{', '.join(c['dropped']) or 'none'}; fallbacks: {', '.join(r.fallbacks()) or 'none'}",
            "",
            "```text",
            r.safe_text if r.ok else "FAILED: " + "; ".join(r.errors),
            "```",
        ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
