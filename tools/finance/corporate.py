"""Person C: corporate finance. Fills data/finance/corporate/.

    python tools/finance/corporate.py --limit 400

Sources (all real, public; no data written by us):
* Enron emails: Hugging Face ``corbt/enron-emails`` (rows API, columns from / to / subject /
  body / date). Keep finance-related mail (trades, credit, budgets, deals); the header
  addresses, names in the signature and phone numbers in the body are the labelled private
  values (found by their format, not by a model).
* TAT-DQA (CC BY 4.0): https://huggingface.co/datasets/next-tat/TAT-DQA, real annual-report
  pages with 16k question-answer pairs: page text as documents, their questions as utility
  questions (the answers are the references).
* Kleister-Charity (real UK charity reports; already used in data/ocr_real, fetcher code in
  tools/fetch_real_testset.py): typed and scanned pages with key fields (charity name,
  income, address, report date) -> the end-to-end scans.

What to write (docs/finance_plan.md, "Contracts"; use ``common.Folder``):
1. emails -> ``f.doc(header + body, private=[addresses, names, phones])``, a third held out;
2. Target Info -> ``f.targeted("What is the phone number of the person who ...?", values=[...])``;
3. TAT-DQA pages -> ``f.doc(page_text)`` and ``f.utility(question, reference=answer,
   doc_text=page_text)`` (questions about figures, never naming the company);
4. 20 Kleister pages -> ``f.scan(pdf_bytes, name, type="annual_report", fields={...},
   private=[the people / address fields], questions=[income, spending, year end],
   text=<the page's text layer, typed pages only>)``.
"""

from __future__ import annotations

import argparse

from common import Folder


def fetch_enron(f: Folder, limit: int) -> None:
    raise NotImplementedError("TODO (Person C): Enron emails -> f.doc / f.targeted")


def fetch_tatdqa(f: Folder, limit: int) -> None:
    raise NotImplementedError("TODO (Person C): TAT-DQA pages + questions -> f.doc / f.utility")


def fetch_kleister(f: Folder, limit: int) -> None:
    raise NotImplementedError("TODO (Person C): Kleister pages -> f.scan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    f = Folder("corporate")
    f.reset()
    fetch_enron(f, args.limit)
    fetch_tatdqa(f, args.limit // 2)
    fetch_kleister(f, 20)
    f.save()


if __name__ == "__main__":
    main()
