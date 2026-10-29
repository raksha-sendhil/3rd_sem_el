"""A smaller corpus for a faster build, always whole people (all of a person's documents or none).

    python tools/corpus_subset.py --docs 150    # -> data/corpus/subset_150.jsonl

* Large enough for everyone the 251 Target Info queries and the 9 attack scripts are about (74
  documents): all of them, then other people (decoys) in person-id order.
* Smaller (a quick demo build): the 3 attack targets first, then their look-alikes (people who
  share a town, district, condition or occupation group with a target, so a stepwise attack
  still narrows a pool), then the other queried people, then anyone else. Target Info then asks
  only the queries about people in the store.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Left out of quick demo builds only (below 74 documents): their third-person leave-request
# email reads wrongly after the rewrite. Builds for the metric are unchanged.
DEMO_LEAVE_OUT = {"p021"}
CORPUS = ROOT / "data" / "corpus"
ATTACKS = ROOT / "data" / "attacks"


def _traits(p: dict) -> set[tuple[str, str]]:
    traits = {(k, p[k]) for k in ("town", "district", "occupation_group")}
    return traits | {("condition", c) for c in p["conditions"]}


def choose(people: list[dict], queried: set[str], targets: set[str], size: int) -> list[dict]:
    named = queried | targets
    if sum(len(p["doc_ids"]) for p in people if p["person_id"] in named) <= size:
        order = [p for p in people if p["person_id"] in named]
        order += [p for p in people if p["person_id"] not in named]
    else:
        target_traits = set().union(*(_traits(p) for p in people if p["person_id"] in targets))

        def rank(p: dict) -> tuple[int, int]:
            if p["person_id"] in targets:
                return (0, 0)
            shared = len(_traits(p) & target_traits)
            return (1, -shared) if shared else (2 if p["person_id"] in queried else 3, 0)

        order = sorted(
            (p for p in people if p["person_id"] not in DEMO_LEAVE_OUT), key=rank
        )  # stable: person-id order within a rank
    chosen, total = [], 0
    for p in order:
        if total + len(p["doc_ids"]) <= size:
            chosen.append(p)
            total += len(p["doc_ids"])
    return chosen


def main(size: int) -> None:
    people = json.loads((CORPUS / "people.json").read_text(encoding="utf-8"))
    queries = json.loads((ATTACKS / "target_info_queries.json").read_text(encoding="utf-8"))
    targets = {
        json.loads(path.read_text(encoding="utf-8"))["target_person_id"]
        for path in ATTACKS.glob("*_p*.json")
    }
    chosen = choose(people, {q["person_id"] for q in queries}, targets, size)
    ids = {d for p in chosen for d in p["doc_ids"]}
    out = CORPUS / f"subset_{size}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for path in sorted((CORPUS / "docs").glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc["doc_id"] in ids:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    kept = sum(p["person_id"] in targets for p in chosen)
    print(f"{len(ids)} documents of {len(chosen)} people ({kept} of 3 attack targets) -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs", type=int, default=150)
    main(parser.parse_args().docs)
