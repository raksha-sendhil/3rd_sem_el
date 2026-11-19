"""Person C: handwritten cheques. Fills data/finance/cheques/ (end-to-end OCR only).

    python tools/finance/cheques.py --zip /kaggle/input/<dataset>/<file>.zip

Source (real): IDRBT Cheque Image Dataset, 112 cheques from four Indian banks, handwritten,
scanned at 300 dpi: https://www.idrbt.ac.in/idrbt-cheque-image-dataset/ (cite Dansena, Bag and
Pal, 2017).

* The site blocks scripts: download the zip once in a browser and upload it as a PRIVATE Kaggle
  dataset; pass its path with ``--zip``. The images are never committed (the dataset has no
  stated licence); ``scans/`` and ``gold.jsonl`` are in .gitignore and are rebuilt on Kaggle.
* The field values are in ``data/finance/cheques/labels.csv``, one row per cheque:
  ``file,bank,payee,payee_2,amount,amount_words,account_number,date``. ``file`` is the image's name in
  the zip (folders ignored). Only labelled cheques are used.
* ``--make-labels`` fills payee, amount and amount in words from the dataset's own
  transcription, ``DATA.xlsx`` in the zip. Two people wrote on each cheque: writer 2 first,
  then writer 1 ("one lakh" + "twenty five thousand", "1" + "25000" -> "one lakh twenty five
  thousand", "1,25,000"), so the amounts are writer 2's part followed by writer 1's. The payee's
  two parts are written in either order, so they are two fields: ``payee`` (writer 1's part)
  and ``payee_2`` (writer 2's), each exactly as written. The account number and date are printed on the
  cheque but not in ``DATA.xlsx``: type them into the CSV by hand (rows already typed are
  kept when the file is made again).

Each labelled cheque -> ``f.scan(image, name, type="cheque", fields={...},
private=["payee", "account_number"], questions=[the amount])``. No ``text`` (no transcription):
the report scores OCR by how many field values it read.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import re
import sys
import zipfile
from itertools import zip_longest
from pathlib import PurePosixPath

from common import FINANCE, Folder

LABELS = FINANCE / "cheques" / "labels.csv"
FIELDS = ["payee", "payee_2", "amount", "amount_words", "account_number", "date"]
COLUMNS = ["file", "bank", *FIELDS]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def read_xlsx(data: bytes) -> list[list[str]]:
    """The first sheet of an .xlsx as rows of strings (no extra packages needed)."""
    x = zipfile.ZipFile(io.BytesIO(data))
    shared = [
        html.unescape(re.sub(r"<[^>]+>", "", s))
        for s in re.findall(r"<si>(.*?)</si>", x.read("xl/sharedStrings.xml").decode("utf-8"), re.S)
    ]
    rows = []
    for row in re.findall(
        r"<row[^>]*>(.*?)</row>", x.read("xl/worksheets/sheet1.xml").decode("utf-8"), re.S
    ):
        cells = []
        for attrs, body in re.findall(r"<c ([^>]*?)(?:/>|>(.*?)</c>)", row, re.S):
            v = re.search(r"<v>(.*?)</v>", body or "")
            value = v.group(1) if v else ""
            cells.append(shared[int(value)] if 't="s"' in attrs and value else html.unescape(value))
        rows.append([c.strip() for c in cells])
    return rows


def indian(digits: str) -> str:
    """'4420000' -> '44,20,000', the grouping the writers use on the cheques."""
    digits = digits.lstrip("0") or "0"
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join([head, *groups, tail]) if head else ",".join([*groups, tail])


def dataset_labels(z: zipfile.ZipFile, images: dict[str, str]) -> list[dict]:
    """payee, amount and amount in words for every cheque, from DATA.xlsx (writer 2's part, then
    writer 1's), ordered so that the first rows take the banks in turn."""
    name = next((n for n in z.namelist() if n.lower().endswith("data.xlsx")), None)
    if not name:
        sys.exit("no DATA.xlsx in the zip")
    rows = read_xlsx(z.read(name))
    by_bank: dict[str, list[dict]] = {}
    skipped = []
    for r in rows[1:]:  # header: CHEQUE NO, USER 1, PEN1, LETTERS WRITTEN 1, USER 2, PEN 2, ...
        if len(r) < 8 or not r[0]:
            continue
        number, first, second, bank = r[0], r[3], r[6], r[7]
        file = next((images[k] for k in images if PurePosixPath(k).stem.endswith(number)), None)
        p1, p2 = [s.strip() for s in first.split(",")], [s.strip() for s in second.split(",")]
        if not file or len(p1) != 3 or len(p2) != 3 or not (p1[2] + p2[2]).isdigit():
            skipped.append(number)
            continue
        by_bank.setdefault(bank, []).append({
            "file": PurePosixPath(file).name, "bank": bank,
            "payee": " ".join(p1[0].split()).title(),
            "payee_2": " ".join(p2[0].split()).title(),
            "amount_words": " ".join(f"{p2[1]} {p1[1]}".split()).lower(),
            "amount": indian(p2[2] + p1[2]),
        })  # fmt: skip
    if skipped:
        print(f"DATA.xlsx: {len(skipped)} rows not in the usual form, left out: {skipped}")
    turns = zip_longest(*by_bank.values())
    return [row for group in turns for row in group if row]


def make_labels(zip_path: str) -> None:
    old = {}
    if LABELS.exists():
        with LABELS.open(encoding="utf-8-sig", newline="") as fh:
            old = {r["file"]: r for r in csv.DictReader(fh) if r.get("file")}
    with zipfile.ZipFile(zip_path) as z:
        images = {n: n for n in z.namelist() if PurePosixPath(n).suffix.lower() in IMAGE_SUFFIXES}
        rows = dataset_labels(z, images)
    for r in rows:  # keep what was typed by hand
        for k in ("account_number", "date"):
            r[k] = (old.get(r["file"], {}).get(k) or "").strip()
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with LABELS.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    banks = sorted({r["bank"] for r in rows})
    print(f"{len(rows)} cheques -> {LABELS} ({', '.join(banks)}); "
          f"{sum(bool(r['account_number']) for r in rows)} with an account number typed")  # fmt: skip


def read_labels() -> list[dict]:
    if not LABELS.exists():
        sys.exit(f"no labels: type the cheques into {LABELS} (columns: file,{','.join(FIELDS)})")
    with LABELS.open(encoding="utf-8-sig", newline="") as fh:
        rows = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]
    rows = [r for r in rows if r.get("file") and any(r.get(k) for k in FIELDS)]
    missing = [r["file"] for r in rows if not (r.get("payee") and r.get("amount"))]
    if missing:
        print(f"warning: no payee or amount for {missing}; those fields are left out")
    return rows


def fetch_cheques(f: Folder, zip_path: str, limit: int) -> None:
    labels = read_labels()
    with zipfile.ZipFile(zip_path) as z:
        images = {PurePosixPath(n).name.lower(): n for n in z.namelist()
                  if PurePosixPath(n).suffix.lower() in IMAGE_SUFFIXES}  # fmt: skip
        print(f"{len(images)} images in the zip, {len(labels)} labelled cheques")
        used = 0
        for row in labels:
            if used == limit:
                break
            name = images.get(PurePosixPath(row["file"]).name.lower())
            if not name:
                print(f"warning: {row['file']} is not in the zip; skipped")
                continue
            fields = {k: row[k] for k in FIELDS if row.get(k)}
            questions = (
                [{"question": "What amount is the cheque written for?", "answer": fields["amount"]}]
                if fields.get("amount")
                else []
            )
            # a bare initial ("J.") is not private: it would "leak" wherever a J. appears
            private = [k for k in ("payee", "payee_2", "account_number")
                       if len(re.sub(r"[^A-Za-z0-9]", "", fields.get(k, ""))) >= 3]  # fmt: skip
            f.scan(z.read(name), PurePosixPath(name).name, type="cheque", fields=fields,
                   private=private, questions=questions)  # fmt: skip
            used += 1
    print(f"cheques: {used} labelled cheques written")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", required=True, help="the IDRBT zip (a private Kaggle dataset)")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--list", action="store_true", help="only list the images in the zip")
    ap.add_argument("--make-labels", action="store_true",
                    help="write labels.csv from the zip's DATA.xlsx (typed columns are kept)")  # fmt: skip
    args = ap.parse_args()
    if args.make_labels:
        make_labels(args.zip)
        return
    if args.list:
        with zipfile.ZipFile(args.zip) as z:
            for n in z.namelist():
                if PurePosixPath(n).suffix.lower() in IMAGE_SUFFIXES:
                    print(PurePosixPath(n).name)
        return
    f = Folder("cheques")
    f.reset()
    fetch_cheques(f, args.zip, args.limit)
    f.save()


if __name__ == "__main__":
    main()
