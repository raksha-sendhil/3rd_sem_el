"""Person B: loan / KYC files. Fills data/finance/kyc/.

    python tools/finance/kyc.py --limit 400

Sources (external synthetic, labelled by their makers; no data written by us):
* Gretel synthetic PII finance (Apache-2.0): https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual
  English split (``data/English_train-*.parquet``): ``generated_text`` with ``pii_spans``
  (every private value with its position) and a document type (loan application, bank
  statement, KYC form, ...). Keep loan / KYC / statement types.
* NIST Special Database 2, filled IRS 1040 tax forms (US government, free):
  https://www.nist.gov/srd/nist-special-database-2 (zip). Each image has an answer file with
  the field values (name, SSN, address, income lines): the end-to-end scans.

What to write (docs/finance_plan.md, "Contracts"; use ``common.Folder``):
1. Gretel documents -> ``f.doc(text, private=[the pii span values])`` (about a third with
   ``heldout=True``), each <= 1,900 characters;
2. Target Info questions -> ``f.targeted(question, values=[...])``: ask for one labelled value
   by describing the document without it ("What is the PAN on the loan application of the
   applicant earning 92,000 a month?");
3. utility questions about content (amount, term, income, product) -> ``f.utility``;
4. 20-30 NIST forms -> ``f.scan(image_bytes, name, type="tax_form", fields={...},
   private=[name, ssn, address fields], questions=[{question, answer}] about income lines)``.

Person B also owns the pipeline (src/sanitise_finance): Gretel's labels are what Agent 1 is
tuned against (precision / recall per category).
"""

from __future__ import annotations

import argparse

from common import Folder


def fetch_gretel(f: Folder, limit: int) -> None:
    raise NotImplementedError("TODO (Person B): Gretel documents -> f.doc / f.targeted / f.utility")


def fetch_nist(f: Folder, limit: int) -> None:
    raise NotImplementedError("TODO (Person B): NIST SD2 forms -> f.scan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    f = Folder("kyc")
    f.reset()
    fetch_gretel(f, args.limit)
    fetch_nist(f, 30)
    f.save()


if __name__ == "__main__":
    main()
