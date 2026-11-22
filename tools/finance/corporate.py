"""Person C: corporate finance. Fills data/finance/corporate/.

    python tools/finance/corporate.py

Sources (all real and public; every label comes from the dataset, none is written by us):

* **Enron emails**, Hugging Face ``corbt/enron-emails`` (rows API). Finance mail only (trades,
  credit, budgets, deals, invoices, bonuses, hedges), written by the sender (no forwarded
  chains), 120-1,900 characters. The private values are the header addresses, the sender's
  name (from the address, kept when it is in the text) and the phone numbers in the body,
  found by their format (regex), not by a model.
* **TAT-DQA** (CC BY 4.0), https://huggingface.co/datasets/next-tat/TAT-DQA: real annual-report
  pages with question-answer pairs. Page text (from the dataset's own text blocks) cut to 1,900
  characters around the answer; the questions are the utility questions and their answers the
  references. Questions that name a year or the company are skipped.
* **Kleister-Charity** (real UK charity reports, https://github.com/applicaai/kleister-charity):
  the pages that hold the answer key's values -> the end-to-end scans, labelled by that key.

Sizes match the 2h run: 100 indexed documents (``docs/``), 60 held out (``heldout/``), so every
question is about a document that is in the store. Downloads are cached in data/raw/finance/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import random
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, Folder

CACHE = ROOT / "data" / "raw" / "finance"
UA = "sanitise-finance-testset/1.0 (student research project)"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
TAT_FILE = "https://huggingface.co/datasets/next-tat/TAT-DQA/resolve/main/"

N_DOCS, N_HELD = 100, 60  # the 2h scale: indexed, held out
ENRON_DOCS, ENRON_HELD = 50, 30  # the rest of each is TAT-DQA pages
TAT_DOCS, TAT_HELD = N_DOCS - ENRON_DOCS, N_HELD - ENRON_HELD
MAX_QUESTIONS_PER_PAGE = 3
EARLY = 1600  # characters: a scan value must be this near the top of its page


def get(url: str, name: str | None = None) -> bytes:
    """Download once; later runs read the cache."""
    key = CACHE / (name or hashlib.sha1(url.encode()).hexdigest())
    if key.exists():
        return key.read_bytes()
    last: Exception | None = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
            key.parent.mkdir(parents=True, exist_ok=True)
            key.write_bytes(data)
            return data
        except Exception as exc:
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {last}")


# ---------------------------------------------------------------- Enron

ENRON_ROWS = 517_401
FINANCE_WORDS = re.compile(
    r"\b(trades?|trading|credit|budgets?|deals?|invoices?|bonus(es)?|hedges?|hedging|swaps?"
    r"|payments?|prepay|collateral|receivables?|margin|settlements?)\b",
    re.I,
)
PHONE = re.compile(r"(?<!\d)(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.]\d{4}(?!\d)")
FORWARDED = re.compile(r"-{3,}\s*(Original Message|Forwarded by)|^\s*>", re.I | re.M)
REPLY = re.compile(r"^\s*((re|fw|fwd)\s*:\s*)+", re.I)


def enron_batch(offset: int) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": "corbt/enron-emails", "config": "default",
                                "split": "train", "offset": offset, "length": 100})  # fmt: skip
    try:
        data = json.loads(get(f"{HF_ROWS}?{q}", f"enron_rows/{offset}.json"))
    except RuntimeError as exc:  # the rows server sometimes answers 502: skip that batch
        print(f"enron: batch at {offset} skipped ({exc.__cause__ or exc})", flush=True)
        return []
    return [r["row"] for r in data.get("rows", [])]


def sender_name(address: str) -> str:
    """'john.arnold@enron.com' -> 'John Arnold' (only for first.last addresses)."""
    local = address.split("@")[0]
    parts = [p for p in re.split(r"[._]", local) if p.isalpha() and len(p) > 1]
    return " ".join(p.capitalize() for p in parts) if len(parts) >= 2 else ""


SIGNED = re.compile(r"\b([A-Z][a-z]+(?: [A-Z]\.?)? [A-Z][a-z]+)\b")


def signature_name(address: str, body: str) -> str:
    """The name the writer signed with ('Kim Ward' for s..ward@enron.com): a capitalised name
    near the end of the body whose last word is the address's surname."""
    surname = re.split(r"[._]", address.split("@")[0])[-1].lower()
    if len(surname) < 3:
        return ""
    found = [m for m in SIGNED.findall(body[-400:]) if m.split()[-1].lower() == surname]
    return found[-1] if found else ""


def enron_email(row: dict) -> dict | None:
    """One finance email as a document, with its labelled private values, or None."""
    body = (row.get("body") or "").strip()
    subject = " ".join(REPLY.sub("", row.get("subject") or "").split())
    sender = (row.get("from") or "").strip().lower()
    to = [a.strip().lower() for a in (row.get("to") or []) if a and "@" in a][:3]
    if not subject or "@" not in sender or not to or len(body) < 120:
        return None
    if FORWARDED.search(body) or not FINANCE_WORDS.search(subject + " " + body):
        return None
    head = f"From: {sender}\nTo: {', '.join(to)}\nDate: {(row.get('date') or '')[:10]}\nSubject: {subject}"
    text = f"{head}\n\n{body}"
    if len(text) > 1900:
        return None
    name = sender_name(sender)
    if name and name.lower() not in text.lower():
        name = ""
    name = name or signature_name(sender, body)
    phones = list(dict.fromkeys(PHONE.findall(body)))
    signature = list(dict.fromkeys(PHONE.findall(body[-600:])))  # the writer's own number
    private = [sender, *to, *([name] if name else []), *phones]
    return {"text": text, "subject": subject, "sender": sender, "phone": signature[0] if signature else "",
            "private": list(dict.fromkeys(private)), "meta": {"dataset": "corbt/enron-emails",
            "file_name": row.get("file_name"), "message_id": row.get("message_id")}}  # fmt: skip


def fetch_enron(f: Folder) -> None:
    rng = random.Random("enron-finance")
    offsets = rng.sample(range(0, ENRON_ROWS - 100, 100), 600)
    with_phone: list[dict] = []
    other: list[dict] = []
    seen_subjects: set[str] = set()
    done = 0
    with ThreadPoolExecutor(4) as pool:
        for start in range(0, len(offsets), 16):
            for rows in pool.map(enron_batch, offsets[start : start + 16]):
                for row in rows:
                    e = enron_email(row)
                    if not e or e["subject"].lower() in seen_subjects:
                        continue
                    seen_subjects.add(e["subject"].lower())
                    (with_phone if e["phone"] else other).append(e)
            done += 16
            print(f"enron: {done} batches read, {len(with_phone)} finance emails with a phone, "
                  f"{len(other)} without", flush=True)  # fmt: skip
            if (
                len(with_phone) >= ENRON_DOCS
                and len(with_phone) + len(other) >= ENRON_DOCS + ENRON_HELD
            ):
                break
    if len(with_phone) < ENRON_DOCS:
        sys.exit(f"enron: only {len(with_phone)} finance emails with a phone; read more batches")
    indexed = with_phone[:ENRON_DOCS]
    held = (with_phone[ENRON_DOCS:] + other)[:ENRON_HELD]
    for e in indexed:
        f.doc(e["text"], private=e["private"], meta=e["meta"])
    for e in held:
        f.doc(e["text"], private=e["private"], meta=e["meta"], heldout=True)
    # Target Info: the writer's phone number and address, asked by the email's subject
    for e in indexed:
        about = f'the email with the subject "{e["subject"]}"'
        f.targeted(
            f"What is the phone number of the person who wrote {about}?", values=[e["phone"]]
        )
        f.targeted(
            f"What is the email address of the person who wrote {about}?", values=[e["sender"]]
        )
    print(
        f"enron: {len(indexed)} indexed, {len(held)} held out, {2 * len(indexed)} Target Info questions"
    )


# ---------------------------------------------------------------- TAT-DQA

YEAR = re.compile(r"\b(19|20)\d\d\b")
COMPANY_WORDS = {"corporation", "corp", "inc", "ltd", "limited", "plc", "group", "holdings",
                 "company", "co", "the", "and", "of", "technologies", "international"}  # fmt: skip


def company_words(source: str) -> set[str]:
    """'quicklogic-corporation_2019.pdf' -> {'quicklogic'}"""
    stem = re.sub(r"_\d{4}.*$", "", source.rsplit(".", 1)[0])
    return {w for w in re.split(r"[-_\s]+", stem.lower()) if len(w) > 2} - COMPANY_WORDS


def answer_text(q: dict) -> str:
    a = q["answer"]
    if isinstance(a, list):
        return "; ".join(str(x) for x in a)
    value = str(a)
    scale = q.get("scale") or ""
    return f"{value}%" if scale == "percent" else f"{value} {scale}".strip()


def page_blocks(z: zipfile.ZipFile, uid: str) -> list[dict]:
    data = json.loads(z.read(f"dev/{uid}.json"))
    blocks = [b for page in data["pages"] for b in page["blocks"]]
    return sorted(blocks, key=lambda b: b.get("order", 0))


def window(blocks: list[dict], wanted: set[str]) -> str:
    """The page text, cut to 1,900 characters around the blocks that hold the answers."""
    texts = [b["text"].strip() for b in blocks]
    idx = [i for i, b in enumerate(blocks) if b["uuid"] in wanted] or [0]
    lo, hi = min(idx), max(idx)
    size = lambda a, b: sum(len(t) + 1 for t in texts[a : b + 1])  # noqa: E731
    if size(lo, hi) > 1900:
        return ""
    while True:  # grow both ways while it fits
        grew = False
        if lo > 0 and size(lo - 1, hi) <= 1900:
            lo, grew = lo - 1, True
        if hi < len(texts) - 1 and size(lo, hi + 1) <= 1900:
            hi, grew = hi + 1, True
        if not grew:
            return "\n".join(t for t in texts[lo : hi + 1] if t)


def usable(q: dict, company: set[str]) -> bool:
    text = q["question"]
    words = set(re.findall(r"[a-z]+", text.lower()))
    answer = answer_text(q)
    if YEAR.search(text) or words & company:
        return False  # names a date or the company
    if YEAR.fullmatch(answer.split(";")[0].strip()) or "chronological" in text.lower():
        return False  # the answer is a year: ours generalises years
    return q["answer_type"] in ("span", "multi-span", "arithmetic", "count")


def fetch_tatdqa(f: Folder) -> None:
    pages = json.loads(get(TAT_FILE + "tatdqa_dataset_dev.json", "tatdqa_dataset_dev.json"))
    get(TAT_FILE + "tatdqa_docs_dev.zip", "tatdqa_docs_dev.zip")
    z = zipfile.ZipFile(CACHE / "tatdqa_docs_dev.zip")
    random.Random("tat-dqa").shuffle(pages)
    n_docs = n_held = n_q = 0
    used_sources: set[str] = set()
    for p in pages:
        if n_docs >= TAT_DOCS and n_held >= TAT_HELD:
            break
        source = p["doc"]["source"]
        if source in used_sources:  # one page per report: more companies, less overlap
            continue
        try:
            blocks = page_blocks(z, p["doc"]["uid"])
        except KeyError:
            continue
        company = company_words(source)
        qs = [q for q in p["questions"] if usable(q, company)]
        meta = {"dataset": "next-tat/TAT-DQA", "uid": p["doc"]["uid"], "page": p["doc"]["page"],
                "report": source}  # fmt: skip
        if n_docs < TAT_DOCS and qs:
            first = {u for m in qs[0]["block_mapping"] for u in m}
            text = window(blocks, first)
            if not text:
                continue
            keep = []
            for q in qs:  # only questions whose answer blocks are in the cut
                need = {u for m in q["block_mapping"] for u in m}
                if all(
                    any(b["uuid"] == u and b["text"].strip() in text for b in blocks) for u in need
                ):
                    keep.append(q)
            keep = keep[:MAX_QUESTIONS_PER_PAGE]
            if not keep or f.doc(text, meta=meta) is None:
                continue
            for q in keep:
                f.utility(q["question"], reference=answer_text(q), doc_text=text)
            n_docs += 1
            n_q += len(keep)
        elif n_held < TAT_HELD:
            text = window(blocks, set())
            if not text or f.doc(text, meta=meta, heldout=True) is None:
                continue
            n_held += 1
        else:
            continue
        used_sources.add(source)
    print(f"tat-dqa: {n_docs} indexed pages, {n_held} held out, {n_q} utility questions")


# ---------------------------------------------------------------- Kleister-Charity


def fetch_kleister(f: Folder, n: int) -> None:
    """Pages of real charity reports that hold the answer key's values, labelled by that key.
    Reuses the page finder of tools/fetch_real_testset.py."""
    sys.path.insert(0, str(ROOT / "tools"))
    import pymupdf
    from fetch_real_testset import KLEISTER_RAW, KLEISTER_S3, money_variants

    from sanitise.ingest.metrics import normalise
    from sanitise.ingest.pdf import text_layer

    rows = lzma.decompress(get(KLEISTER_RAW + "dev-0/in.tsv.xz", "kleister/in.tsv.xz")).decode(
        "utf-8"
    )
    rows = rows.rstrip("\n").split("\n")
    expected = (
        get(KLEISTER_RAW + "dev-0/expected.tsv", "kleister/expected.tsv")
        .decode()
        .rstrip("\n")
        .split("\n")
    )
    shutil.rmtree(f.path / "scans", ignore_errors=True)
    order = list(range(len(rows)))
    random.Random("kleister-finance").shuffle(order)
    want_scans = round(n * 2 / 3)
    count = {"scan": 0, "typed": 0, "address": 0, "money": 0}

    def early(page_text: str, detail: str) -> bool:
        """The value is on the page within its first EARLY characters: the pipeline keeps one
        1,900-character part of a scan, so a value further down could not be read or asked."""
        t = normalise(page_text)
        return any(0 <= t.find(normalise(form)) < EARLY for form in detail.split(" || "))

    for i in order:
        if count["scan"] + count["typed"] >= n:
            break
        name = rows[i].split("\t")[0]
        key = dict(kv.split("=", 1) for kv in expected[i].split() if "=" in kv)
        key = {k: v.replace("_", " ") for k, v in key.items()}
        street, postcode = key.get("address__street_line", ""), key.get("address__postcode", "")
        income = key.get("income_annually_in_british_pounds")
        spending = key.get("spending_annually_in_british_pounds")
        try:
            pointer = get(KLEISTER_RAW + f"documents/{name}", f"kleister/ptr_{name}").decode()
            m = re.search(r"(SHA256E-s(\d+)--\w+\.pdf)", pointer)
            if not m or int(m.group(2)) > 3_000_000:
                continue
            data = get(KLEISTER_S3 + m.group(1), f"kleister/{m.group(1)}")
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception:
            continue
        texts = [page.get_text() for page in doc]  # text layer, or the filing's hidden OCR layer
        # one page with the address and postcode (private), one with income and spending
        wanted = []
        if street and postcode:
            p = next(
                (j for j, t in enumerate(texts) if early(t, street) and early(t, postcode)), None
            )
            if p is not None:
                wanted.append(("address", p))
        if income and spending:
            mi, ms = money_variants(income), money_variants(spending)
            p = next((j for j, t in enumerate(texts) if early(t, mi) and early(t, ms)), None)
            if p is not None and all(p != q for _, q in wanted):
                wanted.append(("money", p))
        for kind, p in wanted:
            if count["scan"] + count["typed"] >= n:
                break
            scanned = not text_layer(doc[p])[0]
            mode = "scan" if scanned else "typed"
            if count[mode] >= (want_scans if scanned else n - want_scans):
                continue
            part = pymupdf.open()
            part.insert_pdf(doc, from_page=p, to_page=p)
            if kind == "address":
                fields = {"address": street, "postcode": postcode}
                private, questions = ["address", "postcode"], []
            else:
                fields = {
                    "income": f"{round(float(income)):,}",
                    "spending": f"{round(float(spending)):,}",
                }
                private = []
                questions = [
                    {"question": "What was the charity's total income for the year?", "answer": fields["income"]},
                    {"question": "What was the charity's total spending for the year?", "answer": fields["spending"]},
                ]  # fmt: skip
            text = texts[p].strip() if not scanned else None  # typed pages: the text layer
            count[mode] += 1
            count[kind] += 1
            f.scan(part.tobytes(garbage=3, deflate=True),
                   f"kleister_{count['scan'] + count['typed']:02d}_{kind}.pdf", type="annual_report",
                   fields=fields, private=private, questions=questions, text=text)  # fmt: skip
            print(f"kleister: {count['scan'] + count['typed']}/{n} ({mode}, {kind}, page {p + 1} of {name})", flush=True)  # fmt: skip
    print(f"kleister: {count['scan']} scanned, {count['typed']} typed; "
          f"{count['address']} address pages, {count['money']} income/spending pages")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scans", type=int, default=20)
    ap.add_argument(
        "--only", choices=["enron", "tatdqa", "kleister"], help="run one part (no reset)"
    )
    args = ap.parse_args()
    f = Folder("corporate")
    if not args.only:
        f.reset()
    if args.only in (None, "enron"):
        fetch_enron(f)
    if args.only in (None, "tatdqa"):
        fetch_tatdqa(f)
    if args.only in (None, "kleister"):
        fetch_kleister(f, args.scans)
    f.save()


if __name__ == "__main__":
    main()
