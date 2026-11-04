"""Measure the PDF/image input step on the OCR test set (and on real handwriting).

    python tools/ocr_eval.py                          # every document in data/ocr_test
    python tools/ocr_eval.py --n 9                    # 9 documents, one of each type
    python tools/ocr_eval.py --types prescription_hw,consent_form_hw --n 4
    python tools/ocr_eval.py --ids lab_photo_02,mixed_file_01
    python tools/ocr_eval.py --n 0 --iam 50           # only 50 real handwritten lines (IAM)
    python tools/ocr_eval.py --no-vision --types discharge_e,lab_e,bank_statement_e  # no GPU

Documents are picked one type at a time (type A 01, type B 01, ..., type A 02, ...), so a small
--n still covers every type. Each document is read once with header removal off; the score is
then computed both with and without header/footer removal on the same reading. Readings are
saved in <out>/pred/, so a run stopped half way (a Kaggle disconnect) carries on where it
stopped; --fresh reads everything again.

IAM lines (--iam N) are real handwritten English sentences with transcriptions (IAM
Handwriting Database, test split, via the Hugging Face dataset Teklia/IAM-line), fetched over
the internet: the synthetic set's handwriting fonts are neater than real hands, so this is the
honest handwriting number.

Writes <out>/report.md (tables), report.json (all numbers) and pred/<id>.txt (what was read).
Headline metric: CER, the character error rate (see sanitise/ingest/metrics.py).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sanitise.config import get_settings  # noqa: E402
from sanitise.ingest import IngestResult, VisionReader, ingest_bytes  # noqa: E402
from sanitise.ingest.headers import strip_repeated  # noqa: E402
from sanitise.ingest.metrics import detail_hits, score_text, strip_scores  # noqa: E402

GROUPS = {
    "electronic PDF": ["discharge_e", "lab_e", "bank_statement_e"],
    "scanned / photographed print": ["discharge_scan", "lab_photo", "affidavit_scan"],
    "handwriting (synthetic)": ["prescription_hw", "consent_form_hw"],
    "mixed file": ["mixed_file"],
}
IAM_ROWS = "https://datasets-server.huggingface.co/rows?dataset=Teklia/IAM-line&config=default&split=test"


def pick(gold: list[dict], n: int | None, types: list[str] | None, ids: list[str] | None) -> list[dict]:
    if ids:
        by_id = {g["id"]: g for g in gold}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"not in the test set: {missing}")
        return [by_id[i] for i in ids]
    pool = [g for g in gold if not types or g["type"] in types]
    order: dict[str, list[dict]] = {}
    for g in pool:
        order.setdefault(g["type"], []).append(g)
    rows = []
    while any(order.values()):
        for t in list(order):
            if order[t]:
                rows.append(order[t].pop(0))
    return rows if n is None else rows[:n]


async def read_doc(path: Path, reader, settings, cache: Path, fresh: bool) -> IngestResult:
    if cache.exists() and not fresh:
        return IngestResult.model_validate_json(cache.read_text(encoding="utf-8"))
    result = await ingest_bytes(
        path.read_bytes(), path.name, reader=reader, settings=settings, strip_headers=False,
        max_chars=10**9,
    )
    cache.write_text(result.model_dump_json(indent=1), encoding="utf-8")
    return result


def score_doc(g: dict, result: IngestResult) -> dict:
    pages = [p.text for p in result.pages]
    stripped, removed = strip_repeated(pages)
    text_strip = "\n\n".join(t for t in stripped if t.strip())
    text_full = "\n\n".join(t for t in pages if t.strip())
    s = score_text(text_strip, g["gold"])
    s_full = score_text(text_full, "\n\n".join(g["gold_pages"]))
    hits = detail_hits(text_strip, g["details"])
    routes = ["vision" if p.method in ("vision", "failed") else "text" for p in result.pages]
    correct, wrong, expected = strip_scores([r.model_dump() for r in removed], g["removals"], text_strip)
    vision_s = [p.seconds for p in result.pages if p.method == "vision"]
    return {
        "id": g["id"],
        "type": g["type"],
        "domain": g["domain"],
        "pages": len(result.pages),
        "failed_pages": sum(p.method == "failed" for p in result.pages),
        "cer": s.cer,
        "wer": s.wer,
        "char_errors": s.char_errors,
        "gold_chars": s.gold_chars,
        "word_errors": s.word_errors,
        "gold_words": s.gold_words,
        "cer_no_strip": s_full.cer,
        "char_errors_no_strip": s_full.char_errors,
        "gold_chars_no_strip": s_full.gold_chars,
        "details": len(hits),
        "details_found": sum(hits),
        "details_missed": [d for d, h in zip(g["details"], hits, strict=True) if not h],
        "routes_ok": sum(a == b for a, b in zip(routes, g["routes"], strict=False)),
        "strip_correct": correct,
        "strip_wrong": wrong,
        "strip_expected": expected,
        "vision_seconds": vision_s,
        "text": text_strip,
    }


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}
    ce = sum(r["char_errors"] for r in rows)
    gc = sum(r["gold_chars"] for r in rows)
    we = sum(r["word_errors"] for r in rows)
    gw = sum(r["gold_words"] for r in rows)
    det = sum(r["details"] for r in rows)
    found = sum(r["details_found"] for r in rows)
    pages = sum(r["pages"] for r in rows)
    vs = [s for r in rows for s in r["vision_seconds"]]
    return {
        "docs": len(rows),
        "pages": pages,
        "cer": ce / max(gc, 1),
        "wer": we / max(gw, 1),
        "detail_recall": found / max(det, 1),
        "details": det,
        "route_accuracy": sum(r["routes_ok"] for r in rows) / max(pages, 1),
        "failed_pages": sum(r["failed_pages"] for r in rows),
        "vision_pages": len(vs),
        "seconds_per_vision_page": sum(vs) / len(vs) if vs else None,
    }


def fetch_iam(n: int, offset: int, cache_dir: Path) -> list[dict]:
    """``n`` IAM test lines: {"idx", "text", "image": path}, cached on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = offset
    while len(rows) < n:
        length = min(100, n - len(rows))
        url = f"{IAM_ROWS}&offset={start}&length={length}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.load(resp)
        batch = data.get("rows", [])
        if not batch:
            break
        for item in batch:
            idx = item["row_idx"]
            img_path = cache_dir / f"iam_{idx}.jpg"
            if not img_path.exists():
                with urllib.request.urlopen(item["row"]["image"]["src"], timeout=60) as resp:
                    img_path.write_bytes(resp.read())
            rows.append({"idx": idx, "text": item["row"]["text"], "image": img_path})
        start += len(batch)
    return rows[:n]


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def report_md(meta: dict, rows: list[dict], iam: list[dict], summary: dict) -> str:
    out = [f"# OCR / document input evaluation - {meta['run']}", ""]
    out.append(f"Vision model `{meta['model']}`, pages rendered at {meta['dpi']} dpi, "
               f"{meta['when']}. Headline: **CER** (character error rate, lower is better; "
               "2% = 2 wrong characters in 100), after header/footer removal.")
    out.append("")
    if rows:
        a = summary["all"]
        out += ["## Overall", "", "| Documents | Pages | CER | WER | Private details read exactly | "
                "Pages routed correctly | Failed pages | s / vision page |",
                "|---|---|---|---|---|---|---|---|",
                f"| {a['docs']} | {a['pages']} | **{pct(a['cer'])}** | {pct(a['wer'])} | "
                f"{pct(a['detail_recall'])} of {a['details']} | {pct(a['route_accuracy'])} | "
                f"{a['failed_pages']} | {a['seconds_per_vision_page'] or 0:.1f} |", ""]
        out += ["## By kind of input", "", "| Kind | Docs | CER | WER | Details read exactly |",
                "|---|---|---|---|---|"]
        for name, s in summary["groups"].items():
            out.append(f"| {name} | {s['docs']} | {pct(s['cer'])} | {pct(s['wer'])} | "
                       f"{pct(s['detail_recall'])} |")
        out += ["", "## By document type", "",
                "| Type | Domain | Docs | Pages | CER | WER | Details | s / vision page |",
                "|---|---|---|---|---|---|---|---|"]
        for t, s in summary["types"].items():
            dom = next(r["domain"] for r in rows if r["type"] == t)
            sec = f"{s['seconds_per_vision_page']:.1f}" if s["seconds_per_vision_page"] else "-"
            out.append(f"| {t} | {dom} | {s['docs']} | {s['pages']} | {pct(s['cer'])} | "
                       f"{pct(s['wer'])} | {pct(s['detail_recall'])} | {sec} |")
        st = summary["strip"]
        out += ["", "## Header/footer removal", "",
                f"On the {st['docs']} multi-page documents with repeated headers/footers: "
                f"{st['correct']} of {st['expected']} repeated lines removed (recall "
                f"{pct(st['recall'])}); {st['wrong']} removed lines whose content is not kept "
                f"anywhere else (information lost; should be 0). CER on these documents: "
                f"{pct(st['cer_strip'])} with removal vs {pct(st['cer_no_strip'])} without.", ""]
        worst = sorted(rows, key=lambda r: -r["cer"])[:5]
        out += ["## Worst documents", "", "| Document | CER | Details missed |", "|---|---|---|"]
        for r in worst:
            missed = ", ".join(r["details_missed"][:4]) or "-"
            out.append(f"| {r['id']} | {pct(r['cer'])} | {missed} |")
        out.append("")
    if iam:
        s = summary["iam"]
        out += ["## Real handwriting (IAM lines)", "",
                f"{s['lines']} handwritten lines from the IAM test split: **CER {pct(s['cer'])}**, "
                f"WER {pct(s['wer'])}, {s['seconds']:.1f} s per line.", "",
                "| Gold | Read | CER |", "|---|---|---|"]
        for r in sorted(iam, key=lambda r: -r["cer"])[:5]:
            out.append(f"| {r['gold']} | {r['read'].replace(chr(10), ' ')} | {pct(r['cer'])} |")
        out.append("")
    return "\n".join(out)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=str(ROOT / "data" / "ocr_test"))
    ap.add_argument("--n", type=int, default=None, help="number of documents (default: all)")
    ap.add_argument("--types", default="", help="comma-separated document types")
    ap.add_argument("--ids", default="", help="comma-separated document ids")
    ap.add_argument("--iam", type=int, default=0, help="real handwritten lines to add (IAM)")
    ap.add_argument("--iam-offset", type=int, default=0)
    ap.add_argument("--model", default=None, help="vision model (default: settings.ocr_model)")
    ap.add_argument("--no-vision", action="store_true", help="no model: scanned pages fail")
    ap.add_argument("--out", default=None, help="default: results/ocr/<model>")
    ap.add_argument("--quiet", action="store_true", help="progress only, not the report text")
    ap.add_argument("--fresh", action="store_true", help="read every document again")
    args = ap.parse_args()

    settings = get_settings()
    model = args.model or settings.ocr_model
    run = model.replace(":", "_").replace("/", "_") if not args.no_vision else "no_vision"
    out = Path(args.out or ROOT / "results" / "ocr" / run)
    (out / "pred").mkdir(parents=True, exist_ok=True)
    test_set = Path(args.set)
    gold = [json.loads(line) for line in (test_set / "gold.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = pick(gold, args.n, [t for t in args.types.split(",") if t] or None,
                  [i for i in args.ids.split(",") if i] or None)

    reader = None if args.no_vision else VisionReader.from_settings(settings, model=model)
    rows: list[dict] = []
    iam_rows: list[dict] = []
    try:
        for k, g in enumerate(chosen, 1):
            started = time.perf_counter()
            result = await read_doc(test_set / g["file"], reader, settings, out / "pred" / f"{g['id']}.json", args.fresh)
            row = score_doc(g, result)
            (out / "pred" / f"{g['id']}.txt").write_text(row.pop("text"), encoding="utf-8")
            rows.append(row)
            print(f"[{k}/{len(chosen)}] {g['id']:22s} CER {pct(row['cer']):>6s}  details "
                  f"{row['details_found']}/{row['details']}  ({time.perf_counter() - started:.0f} s)",
                  flush=True)
        if args.iam:
            from PIL import Image

            from sanitise.ingest import images

            if reader is None:
                raise SystemExit("--iam needs the vision model (drop --no-vision)")
            lines = fetch_iam(args.iam, args.iam_offset, test_set / ".iam")
            for k, line in enumerate(lines, 1):
                cache = out / "pred" / f"iam_{line['idx']}.txt"
                started = time.perf_counter()
                if cache.exists() and not args.fresh:
                    read, secs = cache.read_text(encoding="utf-8"), 0.0
                else:
                    img = Image.open(io.BytesIO(line["image"].read_bytes()))
                    prepared = images.prepare(img, max_side=settings.ocr_max_side, min_side=settings.ocr_min_side)
                    read = await reader.read(prepared)
                    secs = time.perf_counter() - started
                    cache.write_text(read, encoding="utf-8")
                s = score_text(read, line["text"])
                iam_rows.append({"idx": line["idx"], "gold": line["text"], "read": read,
                                 "cer": s.cer, "wer": s.wer, "char_errors": s.char_errors,
                                 "gold_chars": s.gold_chars, "word_errors": s.word_errors,
                                 "gold_words": s.gold_words, "seconds": secs})
                print(f"[IAM {k}/{len(lines)}] CER {pct(s.cer):>6s}", flush=True)
    finally:
        if reader is not None:
            await reader.aclose()

    summary: dict = {"all": summarise(rows)}
    summary["groups"] = {name: summarise([r for r in rows if r["type"] in ts])
                         for name, ts in GROUPS.items() if any(r["type"] in ts for r in rows)}
    summary["types"] = {t: summarise([r for r in rows if r["type"] == t])
                        for t in dict.fromkeys(r["type"] for r in rows)}
    multi = [r for r in rows if r["strip_expected"]]
    c = sum(r["strip_correct"] for r in multi)
    w = sum(r["strip_wrong"] for r in multi)
    e = sum(r["strip_expected"] for r in multi)
    summary["strip"] = {
        "docs": len(multi), "correct": c, "wrong": w, "expected": e,
        "recall": c / e if e else None,
        "cer_strip": sum(r["char_errors"] for r in multi) / max(sum(r["gold_chars"] for r in multi), 1) if multi else None,
        "cer_no_strip": sum(r["char_errors_no_strip"] for r in multi) / max(sum(r["gold_chars_no_strip"] for r in multi), 1) if multi else None,
    }
    if iam_rows:
        timed = [r["seconds"] for r in iam_rows if r["seconds"]]
        summary["iam"] = {
            "lines": len(iam_rows),
            "cer": sum(r["char_errors"] for r in iam_rows) / max(sum(r["gold_chars"] for r in iam_rows), 1),
            "wer": sum(r["word_errors"] for r in iam_rows) / max(sum(r["gold_words"] for r in iam_rows), 1),
            "seconds": sum(timed) / len(timed) if timed else 0.0,
        }
    meta = {"run": out.name, "model": "none" if args.no_vision else model, "dpi": settings.ocr_dpi,
            "when": datetime.now().strftime("%Y-%m-%d %H:%M")}
    (out / "report.json").write_text(json.dumps({"meta": meta, "summary": summary, "docs": rows,
                                                 "iam": iam_rows}, indent=1), encoding="utf-8")
    md = report_md(meta, rows, iam_rows, summary)
    (out / "report.md").write_text(md, encoding="utf-8")
    if not args.quiet:
        print("\n" + md)
    print(f"saved {out / 'report.md'}")


if __name__ == "__main__":
    asyncio.run(main())
