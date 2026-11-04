"""Fetch the real OCR test set: real documents with known text, 50 per domain by default.

    python tools/fetch_real_testset.py                       # 50 health + 50 finance + 50 law
    python tools/fetch_real_testset.py --per-domain 20       # smaller, same mix
    python tools/fetch_real_testset.py --domains health

Writes data/ocr_real/docs/* and data/ocr_real/gold.jsonl (same format as data/ocr_test, plus
``source``, ``licence``, ``url`` and ``order``). Nothing here is committed: the files are fetched
from their public sources each time (downloads are cached in data/ocr_real/.cache), which keeps
every licence's terms and the rule that no real document goes into the repository.

Sources (share of each domain at 50 per domain):

    health   ClinOCR-Bench (40)       clinical documents printed, crumpled, photocopied, put in
                                      plastic sleeves, rotated and scanned; human-audited text;
                                      subsets normal / handwriting / poor / rotated / tables /
                                      mixed. Content is fictional (no patient data). MIT.
             BD prescriptions (10)    real doctors' handwriting: medicine names cut from real
                                      prescriptions in Bangladesh, with their labels.
    finance  SROIE (25)               real scanned receipts and tax invoices (ICDAR 2019).
             CORD (25)                real receipt photos (CC BY 4.0).
    law      CUAD (35)                real contracts filed with the SEC (CC BY 4.0): 2 pages each,
                                      half as the PDF itself, half scanned (rendered and degraded);
                                      most carry a running "Source:" line on every page.
             FUNSD (15)               real noisy scanned business forms and letters from the
                                      tobacco litigation archive (research use).

``order``: "exact" when the gold text comes in reading order (transcriptions, PDF text layers);
"approximate" when it was rebuilt from word boxes (SROIE, CORD, FUNSD). For those, CER also
counts order differences, so the order-free word F1 is the fair number.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from make_ocr_testset import add_image_page, degrade, pdf_bytes, rasterise  # noqa: E402

from sanitise.ingest.headers import strip_repeated  # noqa: E402
from sanitise.ingest.pdf import text_layer  # noqa: E402

UA = "sanitise-ocr-testset/1.0 (student research project)"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
CUAD_TREE = "https://huggingface.co/api/datasets/theatticusproject/cuad/tree/main/"
CUAD_FILE = "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/"

# share of each source in a domain of 50
QUOTAS = {
    "health": {"clinocr": 40, "bd_rx": 10},
    "finance": {"sroie": 25, "cord": 25},
    "law": {"cuad": 35, "funsd": 15},
}
CLINOCR_SUBSETS = {"normal": 7, "handwriting": 8, "poor": 7, "rotated": 6, "tables": 7, "mixed": 5}
ABOUT = {
    "clinocr_normal": "ClinOCR-Bench: clean scan of a clinical document",
    "clinocr_handwriting": "ClinOCR-Bench: clinical document filled in handwriting, printed and scanned",
    "clinocr_poor": "ClinOCR-Bench: low-resolution, photographed or crumpled clinical scan",
    "clinocr_rotated": "ClinOCR-Bench: rotated or skewed clinical scan",
    "clinocr_tables": "ClinOCR-Bench: clinical document dominated by tables",
    "clinocr_mixed": "ClinOCR-Bench: several artifacts at once (rotation, highlight, low resolution)",
    "rx_handwritten_word": "real doctor's handwriting: a medicine name from a real prescription",
    "receipt_scan": "real scanned receipt / tax invoice (SROIE)",
    "receipt_photo": "real receipt photo (CORD)",
    "contract_e": "real contract filed with the SEC, as PDF (CUAD)",
    "contract_scan": "real contract filed with the SEC, scanned (CUAD, rendered and degraded)",
    "form_scan": "real noisy scanned business form or letter (FUNSD)",
}


class Fetcher:
    def __init__(self, cache: Path) -> None:
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)

    def get(self, url: str, *, cache: bool = True, polite: float = 0.0) -> bytes:
        key = self.cache / hashlib.sha1(url.split("?Expires")[0].encode()).hexdigest()
        if cache and key.exists():
            return key.read_bytes()
        last: Exception | None = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                if cache:
                    key.write_bytes(data)
                time.sleep(polite)
                return data
            except Exception as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"could not download {url}: {last}")

    def rows(self, dataset: str, config: str, split: str) -> list[dict]:
        """Every row of a Hugging Face split (image columns come back as URLs)."""
        out: list[dict] = []
        offset = 0
        while True:
            q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                        "offset": offset, "length": 100})
            data = json.loads(self.get(f"{HF_ROWS}?{q}", cache=False))
            batch = [r["row"] for r in data.get("rows", [])]
            out += batch
            offset += len(batch)
            if not batch or offset >= data.get("num_rows_total", 0):
                return out


def spread(total: int, k: int) -> list[int]:
    """``k`` indexes spread evenly over ``total`` (every template, not just the first ones)."""
    k = min(k, total)
    return [int(i * total / k) for i in range(k)]


def scale(quota: dict[str, int], n: int) -> dict[str, int]:
    """Quota for a domain of ``n`` items, same proportions (largest remainder)."""
    total = sum(quota.values())
    raw = {k: v * n / total for k, v in quota.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def save_image(data: bytes, path: Path, max_side: int = 2200) -> None:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    img.save(path, format="JPEG", quality=90)


def lines_from_boxes(items: list[tuple[str, float, float, float, float]]) -> list[str]:
    """Words with boxes (text, x0, y0, x1, y1) to lines: same row = centres within half a line
    height, rows top to bottom, words left to right. Approximate reading order."""
    items = [(t.strip(), *b) for t, *b in items if t and t.strip()]
    if not items:
        return []
    heights = sorted(max(y1 - y0, 1) for _, _, y0, _, y1 in items)
    h = heights[len(heights) // 2]
    lines: list[list] = []
    for it in sorted(items, key=lambda i: (i[2] + i[4]) / 2):
        yc = (it[2] + it[4]) / 2
        if lines and abs(lines[-1][0] - yc) <= 0.5 * h:
            lines[-1][1].append(it)
            lines[-1][0] = sum((a[2] + a[4]) / 2 for a in lines[-1][1]) / len(lines[-1][1])
        else:
            lines.append([yc, [it]])
    return [" ".join(i[0] for i in sorted(line[1], key=lambda i: i[1])) for line in lines]


def clean_page(text: str) -> str:
    lines = [ln.rstrip() for ln in text.replace(chr(160), " ").splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def record(doc_id: str, file: str, doc_type: str, domain: str, gold_pages: list[str],
           routes: list[str], *, source: str, licence: str, url: str, order: str) -> dict:
    stripped, removed = strip_repeated(gold_pages) if len(gold_pages) > 1 else (gold_pages, [])
    return {
        "id": doc_id,
        "file": file,
        "type": doc_type,
        "domain": domain,
        "about": ABOUT[doc_type],
        "routes": routes,
        "gold": "\n\n".join(t for t in stripped if t.strip()),
        "gold_pages": gold_pages,
        "removals": [{"page": r.page, "line": r.line} for r in removed],
        "details": [],
        "source": source,
        "licence": licence,
        "url": url,
        "order": order,
    }


# ------------------------------------------------------------------ sources


def clinocr(f: Fetcher, n: int, out: Path) -> list[dict]:
    recs = []
    for subset, k in scale(CLINOCR_SUBSETS, n).items():
        rows = f.rows("Daniele0025/ClinOCR-Bench", subset, "test")
        for i in spread(len(rows), k):
            r = rows[i]
            doc_id = f"health_clinocr_{r['doc_id']}"
            file = f"docs/{doc_id}.jpg"
            save_image(f.get(r["image"]["src"]), out / file)
            recs.append(record(doc_id, file, f"clinocr_{subset}", "health", [r["ground_truth"]],
                               ["vision"], source="ClinOCR-Bench", licence="MIT",
                               url="https://huggingface.co/datasets/Daniele0025/ClinOCR-Bench",
                               order="exact"))
    return recs


def bd_rx(f: Fetcher, n: int, out: Path) -> list[dict]:
    rows = f.rows("lance-format/handwriting-ocr", "default", "test")
    seen: set[str] = set()
    picked = []
    for r in rows[:: max(1, len(rows) // (n * 4))]:  # different medicines, not 10 of the same
        if r["medicine_name"] not in seen:
            seen.add(r["medicine_name"])
            picked.append(r)
        if len(picked) == n:
            break
    recs = []
    for k, r in enumerate(picked, 1):
        doc_id = f"health_bd_rx_{k:02d}"
        file = f"docs/{doc_id}.jpg"
        save_image(f.get(r["image"]["src"]), out / file)
        recs.append(record(doc_id, file, "rx_handwritten_word", "health", [r["medicine_name"]],
                           ["vision"], source="Doctor's Handwritten Prescription BD dataset",
                           licence="research use (Kaggle mamun1113, via lance-format/handwriting-ocr)",
                           url="https://huggingface.co/datasets/lance-format/handwriting-ocr",
                           order="exact"))
    return recs


def sroie(f: Fetcher, n: int, out: Path) -> list[dict]:
    rows = f.rows("jsdnrs/ICDAR2019-SROIE", "default", "test")
    recs = []
    for k, i in enumerate(spread(len(rows), n), 1):
        r = rows[i]
        doc_id = f"finance_sroie_{k:02d}"
        file = f"docs/{doc_id}.jpg"
        save_image(f.get(r["image"]["src"]), out / file)
        lines = lines_from_boxes([(w, *b) for w, b in zip(r["words"], r["bboxes"], strict=False)])
        recs.append(record(doc_id, file, "receipt_scan", "finance", ["\n".join(lines)], ["vision"],
                           source="SROIE (ICDAR 2019)", licence="research use",
                           url="https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE",
                           order="approximate"))
    return recs


def cord(f: Fetcher, n: int, out: Path) -> list[dict]:
    rows = f.rows("naver-clova-ix/cord-v2", "default", "test")
    recs = []
    for k, i in enumerate(spread(len(rows), n), 1):
        r = rows[i]
        gt = json.loads(r["ground_truth"])
        words = []
        for line in gt.get("valid_line", []):
            for w in line.get("words", []):
                q = w["quad"]
                words.append((w["text"], min(q["x1"], q["x4"]), min(q["y1"], q["y2"]),
                              max(q["x2"], q["x3"]), max(q["y3"], q["y4"])))
        doc_id = f"finance_cord_{k:02d}"
        file = f"docs/{doc_id}.jpg"
        save_image(f.get(r["image"]["src"]), out / file)
        recs.append(record(doc_id, file, "receipt_photo", "finance",
                           ["\n".join(lines_from_boxes(words))], ["vision"], source="CORD v2",
                           licence="CC BY 4.0", url="https://huggingface.co/datasets/naver-clova-ix/cord-v2",
                           order="approximate"))
    return recs


def funsd(f: Fetcher, n: int, out: Path) -> list[dict]:
    rows = f.rows("nielsr/funsd", "default", "test")
    recs = []
    for k, i in enumerate(spread(len(rows), n), 1):
        r = rows[i]
        doc_id = f"law_funsd_{k:02d}"
        file = f"docs/{doc_id}.jpg"
        save_image(f.get(r["image"]["src"]), out / file)
        lines = lines_from_boxes([(w, *b) for w, b in zip(r["words"], r["bboxes"], strict=False)])
        recs.append(record(doc_id, file, "form_scan", "law", ["\n".join(lines)], ["vision"],
                           source="FUNSD", licence="non-commercial research",
                           url="https://guillaumejaume.github.io/FUNSD/", order="approximate"))
    return recs


_BROKEN_QUOTE = re.compile(r"(?:^|[\s(])\?[A-Za-z]|[A-Za-z]\?(?=[)\s,.;:])")
_LOST_LIGATURE = re.compile(r"\b(?:fve|frst|affrmed|fled|specifed|confrm\w*|signifcant\w*|benefts?|fnal\w*)\b")


def gold_ok(text: str) -> bool:
    """A PDF text layer is gold only if it reads as printed: no unknown glyphs (U+FFFD), no
    proof watermark run into the text, no quotes turned into '?', no lost "fi" ligatures."""
    return (
        chr(0xFFFD) not in text
        and "Pending Publication" not in text
        and len(_BROKEN_QUOTE.findall(text)) < 2
        and not _LOST_LIGATURE.search(text)
    )


def pdf_pages_item(data: bytes, pages: list[int], scan: bool, rng: random.Random
                   ) -> tuple[bytes, list[str], list[str]] | None:
    """Two pages of a born-digital PDF: (file, gold text per page, routes); None when a page
    has no trustworthy text layer (then there is no gold)."""
    try:
        src = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        return None
    pages = [p for p in pages if p < src.page_count]
    if len(pages) < 2 or any(not text_layer(src[p])[0] for p in pages):
        return None
    gold = [clean_page(src[p].get_text("text", sort=True)) for p in pages]
    if not all(gold_ok(g) for g in gold):
        return None
    part = pymupdf.open()
    part.insert_pdf(src, from_page=pages[0], to_page=pages[-1])
    if not scan:
        return pdf_bytes(part), gold, ["text"] * len(pages)
    out = pymupdf.open()
    for i in range(part.page_count):
        add_image_page(out, degrade(rasterise(part, i, rng), rng, "scan"), rng)
    return pdf_bytes(out), gold, ["vision"] * len(pages)


def cuad(f: Fetcher, n: int, out: Path) -> list[dict]:
    paths: list[str] = []
    for part in ("Part_I", "Part_II", "Part_III"):
        base = f"CUAD_v1/full_contract_pdf/{part}"
        for folder in json.loads(f.get(CUAD_TREE + urllib.parse.quote(base))):
            if folder["type"] == "directory":
                listing = json.loads(f.get(CUAD_TREE + urllib.parse.quote(folder["path"])))
                paths += sorted(x["path"] for x in listing if x["path"].lower().endswith(".pdf"))
    recs = []
    k = 0
    for i in spread(len(paths), len(paths)):
        if k == n:
            break
        path = paths[(i * 7) % len(paths)]  # a stride through the categories
        scan = k % 2 == 1
        rng = random.Random(f"cuad-{path}")
        item = pdf_pages_item(f.get(CUAD_FILE + urllib.parse.quote(path)), [0, 1], scan, rng)
        if item is None:
            continue
        k += 1
        doc_type = "contract_scan" if scan else "contract_e"
        doc_id = f"law_cuad_{k:02d}"
        file = f"docs/{doc_id}.pdf"
        (out / file).write_bytes(item[0])
        recs.append(record(doc_id, file, doc_type, "law", item[1], item[2], source="CUAD v1",
                           licence="CC BY 4.0", url=CUAD_FILE + urllib.parse.quote(path),
                           order="exact"))
    return recs


SOURCES = {"clinocr": clinocr, "bd_rx": bd_rx, "sroie": sroie, "cord": cord, "cuad": cuad,
           "funsd": funsd}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "ocr_real"))
    ap.add_argument("--per-domain", type=int, default=50)
    ap.add_argument("--domains", default="health,finance,law")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(out / ".cache")
    records: list[dict] = []
    for domain in args.domains.split(","):
        for source, k in scale(QUOTAS[domain], args.per_domain).items():
            if k == 0:
                continue
            print(f"{domain:8s} {source:8s} fetching {k} ...", flush=True)
            got = SOURCES[source](fetcher, k, out)
            print(f"{'':8s} {source:8s} {len(got)} documents", flush=True)
            records += got
    # fetching some domains keeps the others already in gold.jsonl
    domains = set(args.domains.split(","))
    gold_file = out / "gold.jsonl"
    if gold_file.exists():
        kept = [json.loads(line) for line in gold_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        records = [r for r in kept if r["domain"] not in domains] + records
    with open(gold_file, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_domain: dict[str, int] = {}
    for r in records:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
    size = sum((out / r["file"]).stat().st_size for r in records)
    print(f"{len(records)} documents {by_domain}, {size / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
