"""Person C: handwritten cheques. Fills data/finance/cheques/ (end-to-end OCR only).

    python tools/finance/cheques.py --zip /kaggle/input/idrbt-cheques/<file>.zip

Source (real): IDRBT Cheque Image Dataset, 112 cheques from four Indian banks, handwritten,
scanned at 300 dpi: https://www.idrbt.ac.in/idrbt-cheque-image-dataset/
* The site blocks scripts: download the zip once in a browser and upload it as a PRIVATE
  Kaggle dataset; pass its path with ``--zip``. Never commit the images: the dataset has no
  stated licence (research use; cite Dansena, Bag and Pal, 2017).
* Field values: the cheques have no transcriptions. Field boxes for all 112 are released with
  arXiv 2606.20682 (https://arxiv.org/pdf/2606.20682); read payee, amount in figures, amount
  in words, account number and date from those, or type them for 20-30 cheques by hand (that
  is labelling, not generating).

What to write (docs/finance_plan.md, "Contracts"; use ``common.Folder``):
* each cheque -> ``f.scan(image_bytes, name, type="cheque", fields={payee, amount, amount_words,
  account_number, date}, private=["payee", "account_number"],
  questions=[{"question": "What amount is the cheque written for?", "answer": <amount>}])``.
  No ``text`` (no transcription): the report then scores OCR by how many field values it read.
"""

from __future__ import annotations

import argparse

from common import Folder


def fetch_cheques(f: Folder, zip_path: str, limit: int) -> None:
    raise NotImplementedError("TODO (Person C): IDRBT zip + field values -> f.scan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", required=True, help="the IDRBT zip (a private Kaggle dataset)")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()
    f = Folder("cheques")
    f.reset()
    fetch_cheques(f, args.zip, args.limit)
    f.save()


if __name__ == "__main__":
    main()
