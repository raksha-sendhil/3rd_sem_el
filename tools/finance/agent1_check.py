"""Person B: Agent 1's rule pass against Gretel's own labels, found vs missed per label type.

    python tools/finance/agent1_check.py --split train --misses     # for tuning
    python tools/finance/agent1_check.py --split test               # the number to report

Tune on train only: ``--misses`` refuses the test split, which the KYC evaluation runs on.

* **found**: every letter and digit of the labelled value is inside spans Agent 3 removes or
  generalises. A kept span does not count.
* **partly**: some of it is. **missed**: none of it is.
* **amounts changed**: money amounts (a currency sign and a number) that a removed span
  touches. It shows over-sanitising, so it should stay near zero.

Rules only: no model is called, so the counts are exact and repeatable. The LLM pass can only
add to what is found.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict

from common import ROOT
from kyc import NOT_PRIVATE, gretel_rows

sys.path.insert(0, str(ROOT / "src"))

AMOUNT = re.compile(r"[$£€₹]\s?\d[\d,]*(?:\.\d+)?")
TRAIN_ROWS = 1500  # of 50,346: every label occurs, and the rows API limits bigger downloads
MAX_CHARS = 2000  # the pipeline's own limit


def check(rows: list[dict]) -> dict:
    from sanitise_finance.agents.rewriter.routing import route
    from sanitise_finance.rules import rule_detect
    from sanitise_finance.schemas import Action

    counts: dict[str, Counter] = defaultdict(Counter)
    misses: dict[str, list[str]] = defaultdict(list)
    amounts = changed = documents = 0
    for row in rows:
        text = row["text"]
        if len(text) > MAX_CHARS:
            continue
        documents += 1
        covered = [False] * len(text)
        for d in route(rule_detect(text), []):
            if d.action is not Action.KEEP and d.start is not None and d.end is not None:
                covered[d.start : d.end] = [True] * (d.end - d.start)
        for s in row["spans"]:
            if s["label"] in NOT_PRIVATE:
                continue
            at = [i for i in range(s["start"], min(s["end"], len(text))) if text[i].isalnum()]
            if not at:
                continue
            hit = sum(covered[i] for i in at)
            result = "found" if hit == len(at) else "partly" if hit else "missed"
            counts[s["label"]][result] += 1
            if result != "found" and len(misses[s["label"]]) < 20:
                around = " ".join(text[max(0, s["start"] - 35) : s["end"] + 15].split())
                misses[s["label"]].append(f"{result}: {s['value']!r} in ...{around}...")
        for m in AMOUNT.finditer(text):
            amounts += 1
            changed += any(covered[m.start() : m.end()])
    return {
        "documents": documents,
        "labels": {k: dict(v) for k, v in sorted(counts.items())},
        "amounts": amounts,
        "amounts_changed": changed,
        "misses": misses,
    }


def table(result: dict) -> str:
    lines = [f"{result['documents']} documents", "",
             "| label | labelled | found | partly | missed | found % |",
             "|---|--:|--:|--:|--:|--:|"]  # fmt: skip
    total: Counter = Counter()
    for label, c in result["labels"].items():
        n = sum(c.values())
        total.update(c)
        lines.append(f"| {label} | {n} | {c.get('found', 0)} | {c.get('partly', 0)} | "
                     f"{c.get('missed', 0)} | {100 * c.get('found', 0) / n:.1f} |")  # fmt: skip
    n = sum(total.values())
    if n:
        lines.append(f"| **all** | {n} | {total['found']} | {total['partly']} | {total['missed']} "
                     f"| {100 * total['found'] / n:.1f} |")  # fmt: skip
    lines += ["", f"money amounts changed: {result['amounts_changed']} of {result['amounts']}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--misses", action="store_true", help="show missed values (train only)")
    ap.add_argument("--json", help="also write the counts to this file")
    args = ap.parse_args()
    if args.misses and args.split == "test":
        ap.error("misses are only shown on train: never tune on the test split")
    sys.stdout.reconfigure(encoding="utf-8")
    result = check(gretel_rows(args.split, TRAIN_ROWS if args.split == "train" else None))
    misses = result.pop("misses")
    print(table(result))
    if args.misses:
        for label, rows in sorted(misses.items()):
            print(f"\n## {label}")
            for r in rows:
                print(f"  {r}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump({"split": args.split, **result}, out, indent=1)


if __name__ == "__main__":
    main()
