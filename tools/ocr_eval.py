"""Measure the PDF/image input step on the OCR test sets (synthetic and real) and real handwriting.

    python tools/ocr_eval.py                               # every document in every test set
    python tools/ocr_eval.py --domain health               # one domain (health / finance / law)
    python tools/ocr_eval.py --set data/ocr_real --n 30    # 30 real documents, every type
    python tools/ocr_eval.py --set data/ocr_test --n 9     # 9 synthetic, one of each type
    python tools/ocr_eval.py --types clinocr_handwriting,rx_handwritten_word
    python tools/ocr_eval.py --ids lab_photo_02,finance_cord_03
    python tools/ocr_eval.py --n 0 --iam 50                # only 50 real handwritten lines (IAM)
    python tools/ocr_eval.py --no-vision --types discharge_e,lab_e,contract_e   # no GPU

Test sets (``--set``, comma-separated; default both that exist):
    data/ocr_test   synthetic documents with exact text (tools/make_ocr_testset.py)
    data/ocr_real   real documents, 50 per domain (tools/fetch_real_testset.py fetches them)

Documents are picked one type at a time (type A, type B, ..., type A again), so a small --n still
covers every type. Each document is read once with header removal off; the score is computed both
with and without header/footer removal on the same reading. Readings are saved in <out>/pred/, so
a run stopped half way (a Kaggle disconnect) carries on where it stopped; --fresh reads again.

IAM lines (--iam N) are real handwritten English sentences with transcriptions (IAM Handwriting
Database, test split, Hugging Face Teklia/IAM-line), fetched over the internet.

Writes <out>/report.md (tables), report.json (all numbers) and pred/<id>.txt (what was read).
Headline metric: CER, the character error rate; word F1 (order-free) next to it, which is the fair
number for real sources whose gold order is approximate (see sanitise/ingest/metrics.py).
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
# real documents hold characters the Windows console cannot print
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT / "src"))

from sanitise.config import get_settings  # noqa: E402
from sanitise.ingest import IngestResult, VisionReader, ingest_bytes  # noqa: E402
from sanitise.ingest.headers import strip_repeated  # noqa: E402
from sanitise.ingest.metrics import detail_hits, score_text, strip_scores  # noqa: E402

DEFAULT_SETS = [ROOT / "data" / "ocr_test", ROOT / "data" / "ocr_real"]
IAM_ROWS = "https://datasets-server.huggingface.co/rows?dataset=Teklia/IAM-line&config=default&split=test"


def load_sets(spec: str | None) -> list[dict]:
    folders = [Path(s) for s in spec.split(",")] if spec else [d for d in DEFAULT_SETS if (d / "gold.jsonl").exists()]
    gold: list[dict] = []
    for folder in folders:
        gold_file = folder / "gold.jsonl"
        if not gold_file.exists():
            hint = " (run tools/fetch_real_testset.py)" if folder.name == "ocr_real" else ""
            raise SystemExit(f"no {gold_file}{hint}")
        for line in gold_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                g = json.loads(line)
                g["_path"] = str(folder / g["file"])
                g.setdefault("source", "synthetic")
                g.setdefault("order", "exact")
                gold.append(g)
    return gold


def pick(gold: list[dict], n: int | None, types: list[str] | None, ids: list[str] | None,
         domain: str | None) -> list[dict]:
    if ids:
        by_id = {g["id"]: g for g in gold}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"not in the test sets: {missing}")
        return [by_id[i] for i in ids]
    pool = [g for g in gold if (not types or g["type"] in types) and (not domain or g["domain"] == domain)]
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
    has_text = g.get("gold") is not None  # False: scored on its details only (Kleister)
    s = score_text(text_strip, g["gold"] if has_text else "")
    s_full = score_text(text_full, "\n\n".join(g["gold_pages"]) if has_text else "")
    hits = detail_hits(text_strip, g["details"])
    routes = ["vision" if p.method in ("vision", "failed") else "text" for p in result.pages]
    correct, lost, expected = strip_scores([r.model_dump() for r in removed], g["removals"], text_strip)
    vision_s = [p.seconds for p in result.pages if p.method == "vision"]
    return {
        "id": g["id"],
        "type": g["type"],
        "domain": g["domain"],
        "source": g["source"],
        "order": g["order"],
        "pages": len(result.pages),
        "failed_pages": sum(p.method == "failed" for p in result.pages),
        "has_text": has_text,
        "cer": s.cer if has_text else None,
        "wer": s.wer if has_text else None,
        "word_f1": s.word_f1 if has_text else None,
        "char_errors": s.char_errors if has_text else 0,
        "gold_chars": s.gold_chars if has_text else 0,
        "word_errors": s.word_errors if has_text else 0,
        "gold_words": s.gold_words if has_text else 0,
        "words_matched": s.words_matched if has_text else 0,
        "pred_words": s.pred_words if has_text else 0,
        "cer_no_strip": s_full.cer if has_text else None,
        "char_errors_no_strip": s_full.char_errors if has_text else 0,
        "gold_chars_no_strip": s_full.gold_chars if has_text else 0,
        "details": len(hits),
        "details_found": sum(hits),
        "details_missed": [d for d, h in zip(g["details"], hits, strict=True) if not h],
        "routes_ok": sum(a == b for a, b in zip(routes, g["routes"], strict=False)),
        "strip_correct": correct,
        "strip_lost": lost,
        "strip_expected": expected,
        "vision_seconds": vision_s,
        "text": text_strip,
    }


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}
    gw = sum(r["gold_words"] for r in rows)
    texts = [r for r in rows if r.get("has_text", True)]  # documents with a full gold text
    det = sum(r["details"] for r in rows)
    pages = sum(r["pages"] for r in rows)
    vs = [s for r in rows for s in r["vision_seconds"]]
    return {
        "docs": len(rows),
        "pages": pages,
        "cer": sum(r["char_errors"] for r in texts) / max(sum(r["gold_chars"] for r in texts), 1) if texts else None,
        "wer": sum(r["word_errors"] for r in texts) / max(gw, 1) if texts else None,
        "word_f1": 2 * sum(r["words_matched"] for r in texts) / max(sum(r["pred_words"] for r in texts) + gw, 1) if texts else None,
        "detail_recall": sum(r["details_found"] for r in rows) / det if det else None,
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
        with urllib.request.urlopen(f"{IAM_ROWS}&offset={start}&length={length}", timeout=60) as resp:
            batch = json.load(resp).get("rows", [])
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


def table(title: str, groups: dict[str, dict], label: str, extra: dict[str, str] | None = None) -> list[str]:
    out = [f"## {title}", "", f"| {label} | Docs | CER | Word F1 | WER | Details read exactly | s / vision page |",
           "|---|---|---|---|---|---|---|"]
    for name, s in groups.items():
        sec = f"{s['seconds_per_vision_page']:.1f}" if s["seconds_per_vision_page"] else "-"
        tag = f" {extra[name]}" if extra and name in extra else ""
        out.append(f"| {name}{tag} | {s['docs']} | {pct(s['cer'])} | {pct(s['word_f1'])} | "
                   f"{pct(s['wer'])} | {pct(s['detail_recall'])} | {sec} |")
    return [*out, ""]


def report_md(meta: dict, rows: list[dict], iam: list[dict], summary: dict) -> str:
    out = [f"# OCR / document input evaluation - {meta['run']}", ""]
    out.append(f"Vision model `{meta['model']}`, pages rendered at {meta['dpi']} dpi, {meta['when']}. "
               "**CER** = character error rate (lower is better; 2% = 2 wrong characters in 100), "
               "after header/footer removal. **Word F1** = words read correctly regardless of order "
               "(higher is better); use it for sources marked *approx. order*, whose gold text was "
               "rebuilt from word boxes. Details = private details read exactly (synthetic documents, and real "
               "documents scored on their details only, such as the Kleister charity accounts, whose CER "
               "shows as -).")
    out.append("")
    if rows:
        a = summary["all"]
        out += ["## Overall", "", "| Documents | Pages | CER | Word F1 | WER | Details read exactly | "
                "Pages routed correctly | Failed pages | s / vision page |",
                "|---|---|---|---|---|---|---|---|---|",
                f"| {a['docs']} | {a['pages']} | **{pct(a['cer'])}** | {pct(a['word_f1'])} | {pct(a['wer'])} | "
                f"{pct(a['detail_recall'])} | {pct(a['route_accuracy'])} | {a['failed_pages']} | "
                f"{a['seconds_per_vision_page'] or 0:.1f} |", ""]
        out += table("By domain", summary["domains"], "Domain")
        approx = {src: "*(approx. order)*" for src in summary["sources"]
                  if any(r["source"] == src and r["order"] == "approximate" for r in rows)}
        out += table("By source", summary["sources"], "Source", approx)
        out += table("By document type", summary["types"], "Type")
        st = summary["strip"]
        if st["docs"]:
            out += ["## Header/footer removal", "",
                    f"On the {st['docs']} multi-page documents with repeated headers/footers: "
                    f"{st['correct']} of {st['expected']} repeated lines removed (recall {pct(st['recall'])}); "
                    f"{st['lost']} removed lines whose content is not kept anywhere else (information "
                    f"lost; should be 0). CER on these documents: {pct(st['cer_strip'])} with removal vs "
                    f"{pct(st['cer_no_strip'])} without.", ""]
        def badness(r: dict) -> float:
            if r["cer"] is not None:
                return r["cer"]
            return 1 - r["details_found"] / r["details"] if r["details"] else 0.0

        worst = sorted(rows, key=lambda r: -badness(r))[:8]
        out += ["## Worst documents", "", "| Document | Source | CER | Word F1 | Details missed |",
                "|---|---|---|---|---|"]
        for r in worst:
            missed = ", ".join(r["details_missed"][:4]) or "-"
            out.append(f"| {r['id']} | {r['source']} | {pct(r['cer'])} | {pct(r['word_f1'])} | {missed} |")
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
    ap.add_argument("--set", default=None, help="test set folders, comma-separated (default: all)")
    ap.add_argument("--n", type=int, default=None, help="number of documents (default: all)")
    ap.add_argument("--domain", default=None, choices=["health", "finance", "law"])
    ap.add_argument("--types", default="", help="comma-separated document types")
    ap.add_argument("--ids", default="", help="comma-separated document ids")
    ap.add_argument("--iam", type=int, default=0, help="real handwritten lines to add (IAM)")
    ap.add_argument("--iam-offset", type=int, default=0)
    ap.add_argument("--model", default=None, help="vision model (default: settings.ocr_model)")
    ap.add_argument("--no-vision", action="store_true", help="no model: scanned pages fail")
    ap.add_argument("--out", default=None, help="default: results/ocr/<model>")
    ap.add_argument("--fresh", action="store_true", help="read every document again")
    ap.add_argument("--quiet", action="store_true", help="progress only, not the report text")
    args = ap.parse_args()

    settings = get_settings()
    model = args.model or settings.ocr_model
    run = model.replace(":", "_").replace("/", "_") if not args.no_vision else "no_vision"
    out = Path(args.out or ROOT / "results" / "ocr" / run)
    (out / "pred").mkdir(parents=True, exist_ok=True)
    gold = load_sets(args.set)
    chosen = pick(gold, args.n, [t for t in args.types.split(",") if t] or None,
                  [i for i in args.ids.split(",") if i] or None, args.domain)

    reader = None if args.no_vision else VisionReader.from_settings(settings, model=model)
    rows: list[dict] = []
    iam_rows: list[dict] = []
    try:
        for k, g in enumerate(chosen, 1):
            started = time.perf_counter()
            result = await read_doc(Path(g["_path"]), reader, settings, out / "pred" / f"{g['id']}.json", args.fresh)
            row = score_doc(g, result)
            (out / "pred" / f"{g['id']}.txt").write_text(row.pop("text"), encoding="utf-8")
            rows.append(row)
            det = f"details {row['details_found']}/{row['details']}" if row["details"] else ""
            print(f"[{k}/{len(chosen)}] {g['id']:36s} CER {pct(row['cer']):>6s}  word F1 "
                  f"{pct(row['word_f1']):>6s}  {det} ({time.perf_counter() - started:.0f} s)", flush=True)
        if args.iam:
            from PIL import Image

            from sanitise.ingest import images

            if reader is None:
                raise SystemExit("--iam needs the vision model (drop --no-vision)")
            lines = fetch_iam(args.iam, args.iam_offset, ROOT / "data" / "ocr_test" / ".iam")
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

    def group(key: str) -> dict[str, dict]:
        return {v: summarise([r for r in rows if r[key] == v]) for v in dict.fromkeys(r[key] for r in rows)}

    summary: dict = {"all": summarise(rows), "domains": group("domain"), "sources": group("source"),
                     "types": group("type")}
    multi = [r for r in rows if r["strip_expected"]]
    c = sum(r["strip_correct"] for r in multi)
    e = sum(r["strip_expected"] for r in multi)
    summary["strip"] = {
        "docs": len(multi), "correct": c, "lost": sum(r["strip_lost"] for r in multi), "expected": e,
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
