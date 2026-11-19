"""Person A: bank customers. Fills data/finance/bank/ and data/finance/people.json.

    python tools/finance/bank.py --limit 400

Sources (both external, no data written by us):
* CFPB Consumer Complaint Database, real complaint narratives (public, US government):
  Hugging Face mirror ``CFPB/consumer-finance-complaints`` (rows API:
  https://datasets-server.huggingface.co/rows?dataset=CFPB/consumer-finance-complaints&config=default&split=train&offset=0&length=100)
  or the bulk CSV at https://files.consumerfinance.gov/ccdb/complaints.csv.zip.
  Keep rows that have a narrative; CFPB already masks names as "XXXX" (say so in the paper).
* AgamiAI Indian bank statements (synthetic, Apache-2.0): https://huggingface.co/datasets/AgamiAI/Indian-Bank-Statements
  ``train/<type>/<id>.pdf`` + ``<id>.json`` (holder name, address, account number, IFSC,
  branch, transactions). The holders are the population for the multi-question attacks.

What to write (see docs/finance_plan.md, "Contracts"; use ``common.Folder``):
1. complaints -> ``f.doc(text, meta={...})`` (indexed) and ``f.doc(text, heldout=True)``
   (about a third), each <= 1,900 characters (skip longer ones);
2. utility questions about complaint content (``f.utility``), never naming a company,
   person or place; the reference answer copied from the text;
3. statements -> one text document per holder in ``population/`` (``folder="population"``,
   ``person_id=...``): the statement's header and a few transactions as text;
4. ``write_people([...])``: one row per holder: person_id, name, town, district, state,
   bank, branch, account_type, account_number, phone (if any), conditions=[product or
   account type], doc_ids. Leave out fields the source does not have (no age or job).
5. Check the population overlaps: several holders should share state, bank and product,
   or the attacks single everyone out at once. Post the counts on WhatsApp.
"""

from __future__ import annotations

import argparse

from common import Folder, write_people  # noqa: F401


def fetch_complaints(f: Folder, limit: int) -> None:
    raise NotImplementedError("TODO (Person A): CFPB narratives -> f.doc / f.utility")


def fetch_population(f: Folder, limit: int) -> None:
    raise NotImplementedError(
        "TODO (Person A): AgamiAI statements -> population docs + write_people"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    f = Folder("bank")
    f.reset()
    fetch_complaints(f, args.limit)
    fetch_population(f, args.limit)
    f.save()


if __name__ == "__main__":
    main()
