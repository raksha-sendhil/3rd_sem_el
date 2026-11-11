"""Bring the legal pipeline in from the legal branch as its own package, ``sanitise_legal``.

    python tools/import_legal.py                      # from origin/law-pipeline-final
    python tools/import_legal.py --ref <branch-or-commit>

The healthcare pipeline (``src/sanitise``) and the legal one grew apart from the first commit
on, so a line-by-line merge would mix them in the same files. Instead the legal branch's code
is copied as it is into ``src/sanitise_legal`` with only two kinds of change:

* the package name (``sanitise`` -> ``sanitise_legal``) in imports, ``-m`` commands and module
  strings, so both packages live side by side;
* its ontology tables go to ``data/legal/ontology`` (main keeps its own ``data/ontology``).

Its data (``data/legal``), profile (``config/domains.json``), tests, tools, docs and notebooks
come along with the same renames. The API serves it under ``/legal`` (``sanitise.api.app``) and
the dashboard's Healthcare / Legal toggle picks between the two. Run this again to take newer
legal-branch work; ``src/sanitise_legal`` is replaced as a whole, so make legal changes on the
legal branch (or here, then copy them back) rather than in both places.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (prefix on the legal branch, where it goes here); files outside these are not taken
MAP = [
    ("src/sanitise/", "src/sanitise_legal/"),
    ("data/legal/", "data/legal/"),
    ("data/ontology/", "data/legal/ontology/"),
    ("config/", "config/"),
    ("tests/legal/", "tests/legal/"),
    # its shared fixtures, for the legal tests only (main's tests keep main's conftest)
    ("tests/conftest.py", "tests/legal/conftest.py"),
    ("tools/legal/", "tools/legal/"),
    ("docs/legal/", "docs/legal/"),
    ("examples/legal_inputs/", "examples/legal_inputs/"),
    ("notebooks/legal_kaggle.ipynb", "notebooks/legal_kaggle.ipynb"),
    ("notebooks/legal_agents_kaggle.ipynb", "notebooks/legal_agents_kaggle.ipynb"),
]
# whole folders replaced on every import (so a file deleted on the branch goes here too)
REPLACED = ["src/sanitise_legal", "data/legal", "tests/legal", "tools/legal", "docs/legal"]
TEXT = {".py", ".md", ".ipynb", ".toml", ".txt", ".cfg", ".ini", ".yaml", ".yml", ".sh"}
SKIP = re.compile(r"(__pycache__|\.pyc$|/store\.sqlite|/index/|/results/|/processed/|/bundle/)")

RENAMES = [
    (re.compile(r"\bfrom sanitise(?=[.\s])"), "from sanitise_legal"),
    # "import sanitise" as a statement, not "from x import sanitise" (a function of that name)
    (re.compile(r"(?m)^(\s*)import sanitise\b(?!_)"), r"\1import sanitise_legal"),
    (re.compile(r"(?<=[\"'`(])sanitise\.(?=[a-z_])"), "sanitise_legal."),
    (re.compile(r"-m sanitise\b(?!_)"), "-m sanitise_legal"),
    (re.compile(r"\bsrc/sanitise/"), "src/sanitise_legal/"),
    (re.compile(r"data/ontology\b"), "data/legal/ontology"),
    (re.compile(r'/ "data" / "ontology"'), '/ "data" / "legal" / "ontology"'),
    # the legal notebooks clone the repo: after the merge the code is on main
    (re.compile(r"-b (?:a/legal-pipeline|law-pipeline-final|law-version-2)\b"), "-b main"),
]


# Small fixes needed only because the legal code now sits next to main's data; each must match
# exactly (the import stops if the legal branch changed the line, so the fix is looked at again).
PATCHES = [
    (
        "src/sanitise_legal/attacks/runner.py",
        '        attack = AttackDefinition.model_validate_json(path.read_text(encoding="utf-8"))\n',
        '        data = json.loads(path.read_text(encoding="utf-8"))\n'
        "        if not isinstance(data, dict):  # main's data/attacks also holds a query list\n"
        "            continue\n"
        "        attack = AttackDefinition.model_validate(data)\n",
    ),
]


def _patch() -> None:
    for path, old, new in PATCHES:
        p = ROOT / path
        text = p.read_text(encoding="utf-8")
        if new in text:
            continue
        if old not in text:
            raise SystemExit(f"{path}: the line to patch changed on the legal branch: {old!r}")
        p.write_text(text.replace(old, new, 1), encoding="utf-8")


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def _read_all(ref: str, paths: list[str]) -> dict[str, bytes]:
    """Every file's contents in one ``git cat-file --batch`` call."""
    query = "".join(f"{ref}:{p}\n" for p in paths).encode("utf-8")
    out = subprocess.run(["git", "-C", str(ROOT), "cat-file", "--batch"], input=query,
                         capture_output=True, check=True).stdout  # fmt: skip
    blobs, pos = {}, 0
    for path in paths:
        end = out.index(b"\n", pos)
        size = int(out[pos:end].split()[2])
        blobs[path] = out[end + 1 : end + 1 + size]
        pos = end + 1 + size + 1
    return blobs


def _target(path: str) -> str | None:
    for src, dst in MAP:
        if path == src or (src.endswith("/") and path.startswith(src)):
            return dst + path[len(src) :] if src.endswith("/") else dst
    return None


def rename(text: str) -> str:
    for pattern, repl in RENAMES:
        text = pattern.sub(repl, text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="origin/law-pipeline-final")
    args = ap.parse_args()
    commit = _git("rev-parse", "--short", args.ref).decode().strip()
    files = _git("ls-tree", "-r", "--name-only", args.ref).decode().splitlines()
    for folder in REPLACED:
        shutil.rmtree(ROOT / folder, ignore_errors=True)
    wanted = [(p, _target(p)) for p in files]
    wanted = [(p, d) for p, d in wanted if d is not None and not SKIP.search(p)]
    blobs = _read_all(args.ref, [p for p, _ in wanted])
    for path, dst in wanted:
        data = blobs[path]
        if Path(path).suffix in TEXT:
            data = rename(data.decode("utf-8")).encode("utf-8")
        out = ROOT / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    n = len(wanted)
    _patch()
    (ROOT / "src/sanitise_legal/IMPORTED_FROM").write_text(
        f"{args.ref} {commit}\n(tools/import_legal.py; do not edit here, see its docstring)\n",
        encoding="utf-8",
    )
    print(f"imported {n} files from {args.ref} ({commit})")


if __name__ == "__main__":
    main()
