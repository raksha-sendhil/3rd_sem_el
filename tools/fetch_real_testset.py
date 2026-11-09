"""Fetch the real OCR test set: real documents with known text, 50 per domain by default.

    python tools/fetch_real_testset.py                       # 50 health + 50 finance + 50 law
    python tools/fetch_real_testset.py --per-domain 20       # smaller, same mix
    python tools/fetch_real_testset.py --domains health

Writes data/ocr_real/docs/* and data/ocr_real/gold.jsonl (same format as data/ocr_test, plus
``source``, ``licence``, ``url`` and ``order``). Nothing here is committed: the files are fetched
from their public sources each time (downloads are cached in data/ocr_real/.cache), which keeps
every licence's terms and the rule that no real document goes into the repository.

Sources (share of each domain at 50 per domain):

    health   ClinOCR-Bench (30)       clinical documents printed, crumpled, photocopied, put in
                                      plastic sleeves, rotated and scanned; human-audited text;
                                      subsets normal / handwriting / poor / rotated / tables /
                                      mixed. Content is fictional (no patient data). MIT.
             BD prescriptions (20)    real doctors' handwriting: medicine names cut from real
                                      prescriptions in Bangladesh, with their labels; 20 different
                                      medicines spread from A to Z.
    finance  Kleister-Charity (50)    real annual reports and accounts of UK charities, filed with the
                                      Charity Commission (many scanned, some typed): trustees' reports,
                                      statements of financial activities, balance sheets, examiners'
                                      reports. Gold = the values a lender or auditor looks up (charity
                                      name and number, postcode, year end, income, spending), kept only
                                      where they appear in the document; scored on details read
                                      exactly, as there is no full transcription. Each filing is cut
                                      to the (up to 4) pages that hold the answers, so the OCR run
                                      stays short.
    law      CUAD (35)                real contracts filed with the SEC (CC BY 4.0): 2 pages each,
                                      half as the PDF itself, half scanned (rendered and degraded);
                                      most carry a running "Source:" line on every page.
             FUNSD (15)               real noisy scanned business forms and letters from the
                                      tobacco litigation archive (research use).

``order``: "exact" when the gold text comes in reading order (transcriptions, PDF text layers);
"approximate" when it was rebuilt from word boxes (FUNSD). For those, CER also counts order
differences, so the order-free word F1 is the fair number. ``gold`` is null for documents that are
scored on their details only (Kleister-Charity).
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
    "health": {"clinocr": 30, "bd_rx": 20},
    "finance": {"kleister": 50},
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
    "accounts_scan": "real charity annual report and accounts, scanned (Kleister-Charity)",
    "accounts_e": "real charity annual report and accounts, typed PDF (Kleister-Charity)",
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


def record(doc_id: str, file: str, doc_type: str, domain: str, gold_pages: list[str] | None,
           routes: list[str], *, source: str, licence: str, url: str, order: str,
           details: list[str] | None = None) -> dict:
    if gold_pages is None:  # scored on its details only (no full transcription)
        return {
            "id": doc_id, "file": file, "type": doc_type, "domain": domain,
            "about": ABOUT[doc_type], "routes": routes, "gold": None, "gold_pages": None,
            "removals": [], "details": details or [], "source": source, "licence": licence,
            "url": url, "order": order,
        }
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
        "details": details or [],
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
    # one crop per medicine, medicines spread over the whole list (A to Z), and within a
    # medicine a crop from the middle of its samples rather than always the first one
    by_name: dict[str, list[dict]] = {}
    for r in rows:
        by_name.setdefault(r["medicine_name"], []).append(r)
    names = sorted(by_name)
    picked = [by_name[names[i]][len(by_name[names[i]]) // 2] for i in spread(len(names), n)]
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

KLEISTER_RAW = "https://raw.githubusercontent.com/applicaai/kleister-charity/master/"
KLEISTER_S3 = "https://applica-public.s3-eu-west-1.amazonaws.com/kleister-charity-git-annex/"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def money_variants(value: str) -> str:
    """'41721.00' -> '41,721 || 41721' (accounts print whole pounds, with or without commas)."""
    pounds = str(round(float(value)))
    return f"{int(pounds):,} || {pounds}"


def date_variants(iso: str) -> str:
    """'2016-03-31' -> the ways accounts print a year end."""
    y, m, d = iso.split("-")
    month = MONTHS[int(m) - 1]
    day = str(int(d))
    n = int(day)
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    forms = [f"{day} {month} {y}", f"{day}{suffix} {month} {y}", f"{d}/{m}/{y}", f"{d}.{m}.{y}",
             f"{d}-{m}-{y}", f"{day} {month[:3]} {y}", f"{month} {day}, {y}"]
    return " || ".join(dict.fromkeys(forms))


def kleister_details(expected: str, text: str) -> list[str]:
    """The answer key's values, each with the forms it may be printed in, kept only when one of
    the forms is in the document's own text (Kleister's text layer / OCR)."""
    from sanitise.ingest.metrics import detail_hits

    fields = dict(kv.split("=", 1) for kv in expected.split() if "=" in kv)
    out = []
    for key, value in fields.items():
        value = value.replace("_", " ")
        if key == "charity_number":
            out.append(value)
        elif key == "address__postcode":
            out.append(value)
        elif key == "charity_name":
            out.append(value)
        elif key in ("income_annually_in_british_pounds", "spending_annually_in_british_pounds"):
            out.append(money_variants(value))
        elif key == "report_date":
            out.append(date_variants(value))
    return [d for d, hit in zip(out, detail_hits(text, out), strict=True) if hit]


KLEISTER_MAX_PAGES = 4


def kleister_pages(doc: pymupdf.Document, details: list[str]) -> tuple[list[int], list[str]]:
    """The pages (at most KLEISTER_MAX_PAGES, in order) that hold the most answers, and the answers
    on them. Pages are found by their text: the typed text layer, or the hidden OCR layer most
    scanned filings carry (used only to find the page; the pipeline reads the scan itself)."""
    from sanitise.ingest.metrics import detail_hits

    on_page = [detail_hits(page.get_text(), details) for page in doc]
    chosen: list[int] = []
    covered = [False] * len(details)
    while len(chosen) < KLEISTER_MAX_PAGES:
        gain, best = max(
            ((sum(h and not c for h, c in zip(hits, covered, strict=True)), i)
             for i, hits in enumerate(on_page) if i not in chosen),
            default=(0, -1),
        )
        if gain == 0:
            break
        chosen.append(best)
        covered = [c or h for c, h in zip(covered, on_page[best], strict=True)]
    return sorted(chosen), [d for d, c in zip(details, covered, strict=True) if c]


def kleister(f: Fetcher, n: int, out: Path) -> list[dict]:
    import lzma

    rows = lzma.decompress(f.get(KLEISTER_RAW + "dev-0/in.tsv.xz")).decode("utf-8").rstrip(chr(10)).split(chr(10))
    expected = f.get(KLEISTER_RAW + "dev-0/expected.tsv").decode().rstrip(chr(10)).split(chr(10))
    assert len(rows) == len(expected), "Kleister dev-0: inputs and answers do not line up"
    order = list(range(len(rows)))
    random.Random("kleister").shuffle(order)
    want_scans = round(n * 2 / 3)  # a lender's files are mostly scans: about 2 scanned to 1 typed
    recs: list[dict] = []
    scans = typed = 0
    for i in order:
        if len(recs) == n:
            break
        cols = rows[i].split(chr(9))
        name, text = cols[0], cols[-1].replace("\\n", chr(10))
        m = re.search(r"(SHA256E-s(\d+)--\w+\.pdf)", f.get(KLEISTER_RAW + f"documents/{name}").decode())
        if not m or int(m.group(2)) > 3_000_000:
            continue
        try:
            doc = pymupdf.open(stream=f.get(KLEISTER_S3 + m.group(1)), filetype="pdf")
        except Exception:
            continue
        pages, details = kleister_pages(doc, kleister_details(expected[i], text))
        if len(details) < 3:  # too little to score, or the pages cannot be found
            continue
        scanned = sum(not text_layer(doc[p])[0] for p in pages) > len(pages) / 2
        if (scanned and scans >= want_scans) or (not scanned and typed >= n - want_scans):
            continue
        scans += scanned
        typed += not scanned
        part = pymupdf.open()
        for page in pages:
            part.insert_pdf(doc, from_page=page, to_page=page)
        doc_id = f"finance_kleister_{len(recs) + 1:02d}"
        file = f"docs/{doc_id}.pdf"
        (out / file).write_bytes(part.tobytes(garbage=3, deflate=True))
        routes = ["text" if text_layer(page)[0] else "vision" for page in part]
        recs.append(record(doc_id, file, "accounts_scan" if scanned else "accounts_e", "finance",
                           None, routes, source="Kleister-Charity", licence="research use",
                           url=f"https://github.com/applicaai/kleister-charity (pages {', '.join(str(p + 1) for p in pages)} of {name})",
                           order="exact", details=details))
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


SOURCES = {"clinocr": clinocr, "bd_rx": bd_rx, "kleister": kleister, "cuad": cuad,
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
