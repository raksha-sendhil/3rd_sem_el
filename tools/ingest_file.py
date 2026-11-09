"""Read one PDF, image or text file and print what the pipeline gets (optionally sanitise it).

    python tools/ingest_file.py scan.pdf
    python tools/ingest_file.py photo.jpg --sanitise          # then Agents 1-3 on each part
    python tools/ingest_file.py prescription_hw_03            # a test-set document: also scored
    python tools/ingest_file.py https://example.org/report.pdf
    python tools/ingest_file.py report.pdf --no-strip --force-vision --out read.txt

Needs Ollama with the vision model (settings.ocr_model, default qwen3-vl:8b-instruct) for scanned
pages and images; electronic PDFs and text files need no model. --sanitise also needs the
agents' backend (the same environment variables as the API: SANITISE_AGENTS_BACKEND=gemini ...).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# real documents hold characters the Windows console cannot print
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT / "src"))

from sanitise.config import get_settings  # noqa: E402
from sanitise.ingest import VisionReader, ingest_bytes  # noqa: E402
from sanitise.ingest.metrics import detail_hits, score_text  # noqa: E402

TEST_SETS = [ROOT / "data" / "ocr_test", ROOT / "data" / "ocr_real"]


def load(source: str) -> tuple[bytes, str, dict | None]:
    """(bytes, file name, gold record if it is a test-set document)."""
    for test_set in TEST_SETS:
        gold_file = test_set / "gold.jsonl"
        if not gold_file.exists() or Path(source).exists() or "/" in source:
            continue
        for line in gold_file.read_text(encoding="utf-8").splitlines():
            g = json.loads(line)
            if g["id"] == source:
                path = test_set / g["file"]
                return path.read_bytes(), path.name, g
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "sanitise-ingest"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read(), source.rsplit("/", 1)[-1].split("?")[0], None
    path = Path(source)
    if not path.exists():
        raise SystemExit(f"no such file, URL or test-set id: {source}")
    return path.read_bytes(), path.name, None


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="file path, URL, or test-set id (e.g. lab_photo_02)")
    ap.add_argument("--sanitise", action="store_true", help="run Agents 1-3 on the text")
    ap.add_argument("--no-strip", action="store_true", help="keep repeated headers/footers")
    ap.add_argument("--force-vision", action="store_true", help="vision model for every page")
    ap.add_argument("--model", default=None, help="vision model (default: settings.ocr_model)")
    ap.add_argument("--out", default=None, help="also write the text to this file")
    args = ap.parse_args()

    settings = get_settings()
    data, name, gold = load(args.source)
    async with VisionReader.from_settings(settings, model=args.model) as reader:
        result = await ingest_bytes(
            data, name, reader=reader, settings=settings,
            strip_headers=False if args.no_strip else None, force_vision=args.force_vision,
        )

    print(f"{name}: {result.kind}, {len(result.pages)} page(s), {result.seconds:.1f} s")
    for p in result.pages:
        why = f"  ({p.note})" if p.note else ""
        print(f"  page {p.index + 1}: {p.method:6s} {len(p.text):6,d} chars  {p.seconds:5.1f} s{why}")
    if result.removed_lines:
        print(f"  removed {len(result.removed_lines)} repeated header/footer line(s):")
        for r in result.removed_lines:
            print(f"    page {r.page + 1}: {r.line}")
    for w in result.warnings:
        print(f"  ! {w}")
    print("\n" + "=" * 30 + " TEXT " + "=" * 30)
    print(result.text)
    print("=" * 66)
    if args.out:
        Path(args.out).write_text(result.text, encoding="utf-8")
        print(f"written to {args.out}")

    if gold is not None:
        hits = detail_hits(result.text, gold["details"])
        details = f"private details read exactly {sum(hits)}/{len(hits)}" if hits else ""
        if gold.get("gold") is None:  # scored on its details only (no full transcription)
            print(f"\nTest-set document ({gold['about']}): {details}")
        else:
            s = score_text(result.text, gold["gold"] if not args.no_strip else "\n\n".join(gold["gold_pages"]))
            print(f"\nTest-set document ({gold['about']}): CER {100 * s.cer:.1f}%, "
                  f"word F1 {100 * s.word_f1:.1f}%, WER {100 * s.wer:.1f}%" + (f", {details}" if details else ""))
        if gold.get("source"):
            print(f"  source: {gold['source']} ({gold['licence']}), gold order {gold['order']}")
        missed = [d for d, h in zip(gold["details"], hits, strict=True) if not h]
        if missed:
            print("  missed:", "; ".join(missed))

    if args.sanitise:
        from sanitise.agents.pipeline import sanitise
        from sanitise.llm.factory import build_llm_client, close_llm_client
        from sanitise.loader import compute_doc_id

        llm = build_llm_client(settings)
        try:
            for i, part in enumerate(result.parts, 1):
                record = await sanitise(compute_doc_id(part), part, llm)
                print(f"\n{'=' * 24} SANITISED part {i}/{len(result.parts)} {'=' * 24}")
                print(record.rewriter.safe_text)
        finally:
            await close_llm_client(llm)


if __name__ == "__main__":
    asyncio.run(main())
