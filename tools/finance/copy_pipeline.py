"""Start the finance pipeline as a copy of the healthcare one: ``src/sanitise`` -> ``src/sanitise_finance``.

    python tools/finance/copy_pipeline.py            # once, at the start (already done on the branch)
    python tools/finance/copy_pipeline.py --force    # start again from health's current code

Person B then tunes ``src/sanitise_finance`` for finance documents (recognisers, prompts,
ontology ladders); ``src/sanitise`` (healthcare) is never touched. Only two things change in the
copy:

* the package name (``sanitise`` -> ``sanitise_finance``) in imports, ``-m`` commands and module
  strings, so both packages live side by side;
* default data paths: the ontology tables are ``data/finance/ontology`` (seeded with health's
  tables), the store ``data/finance/store.sqlite``, the index ``data/finance/index``, the attacks
  ``data/finance/attacks`` and the person table ``data/finance/people.json``.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC, DST = ROOT / "src" / "sanitise", ROOT / "src" / "sanitise_finance"

RENAMES = [
    (re.compile(r"\bfrom sanitise(?=[.\s])"), "from sanitise_finance"),
    # "import sanitise" as a statement, not "from x import sanitise" (a function of that name)
    (re.compile(r"(?m)^(\s*)import sanitise\b(?!_)"), r"\1import sanitise_finance"),
    (re.compile(r"(?<=[\"'`(])sanitise\.(?=[a-z_])"), "sanitise_finance."),
    (re.compile(r"-m sanitise\b(?!_)"), "-m sanitise_finance"),
    (re.compile(r'/ "data" / "ontology"'), '/ "data" / "finance" / "ontology"'),
    (re.compile(r'"data/store\.sqlite"'), '"data/finance/store.sqlite"'),
    (re.compile(r'"data/index"'), '"data/finance/index"'),
    (re.compile(r'"data/attacks"'), '"data/finance/attacks"'),
    (re.compile(r'"data/corpus/people\.json"'), '"data/finance/people.json"'),
]


def rename(text: str) -> str:
    for pattern, repl in RENAMES:
        text = pattern.sub(repl, text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="replace an existing src/sanitise_finance")
    args = ap.parse_args()
    if DST.exists():
        if not args.force:
            raise SystemExit(f"{DST} exists (Person B's work): pass --force to start it again")
        shutil.rmtree(DST)
    n = 0
    for path in sorted(SRC.rglob("*")):
        if "__pycache__" in path.parts or path.is_dir():
            continue
        out = DST / path.relative_to(SRC)
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in {".py", ".md", ".txt", ".json"}:
            out.write_text(rename(path.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            out.write_bytes(path.read_bytes())
        n += 1
    onto = ROOT / "data" / "finance" / "ontology"
    if not onto.exists():
        shutil.copytree(ROOT / "data" / "ontology", onto)
    (DST / "COPIED_FROM").write_text(
        "src/sanitise (healthcare) via tools/finance/copy_pipeline.py; tune here for finance.\n",
        encoding="utf-8",
    )
    print(
        f"copied {n} files to {DST.relative_to(ROOT)}; ontology seeded in {onto.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
