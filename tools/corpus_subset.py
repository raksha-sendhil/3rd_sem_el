"""A smaller corpus for a faster build: every document of the people the attacks are about, then
whole other people (decoys) in person-id order until the size is reached.

    python tools/corpus_subset.py --docs 150    # -> data/corpus/subset_150.jsonl

Every person the 251 Target Info queries or the 9 attack scripts name keeps all their documents,
so no query asks about someone who is missing; a decoy is taken whole or not at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
ATTACKS = ROOT / "data" / "attacks"


def main(size: int) -> None:
    people = json.loads((CORPUS / "people.json").read_text(encoding="utf-8"))
    queries = json.loads((ATTACKS / "target_info_queries.json").read_text(encoding="utf-8"))
    named = {q["person_id"] for q in queries}
    for path in ATTACKS.glob("*_p*.json"):
        named.add(json.loads(path.read_text(encoding="utf-8"))["target_person_id"])
    chosen = [p for p in people if p["person_id"] in named]
    total = sum(len(p["doc_ids"]) for p in chosen)
    if total > size:
        raise SystemExit(f"the attacked people alone have {total} documents (> {size})")
    for p in people:
        if p["person_id"] not in named and total + len(p["doc_ids"]) <= size:
            chosen.append(p)
            total += len(p["doc_ids"])
    ids = {d for p in chosen for d in p["doc_ids"]}
    out = CORPUS / f"subset_{size}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for path in sorted((CORPUS / "docs").glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc["doc_id"] in ids:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"{len(ids)} documents of {len(chosen)} people ({len(named)} attacked) -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs", type=int, default=150)
    main(parser.parse_args().docs)
