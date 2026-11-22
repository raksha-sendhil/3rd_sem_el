"""The legal and finance demo stores the API serves under /legal and /finance.

    python tools/build_demo_stores.py                      # both, with each domain's own agents
    python tools/build_demo_stores.py --only finance --mock

Healthcare's store is tools/build_store.py's. Each domain is built by its own package, so with
its own prompts, rules and models. Safe to re-run: documents already in a store are skipped.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(domain: str, docs: int, work: Path, store: Path) -> list[str]:
    demo = ROOT / "data" / domain / "demo" / "docs"
    corpus = work / "corpus"
    corpus.mkdir(parents=True)
    for path in sorted(demo.glob("*.json"))[:docs]:
        shutil.copyfile(path, corpus / path.name)
    if domain == "legal":
        shutil.copyfile(ROOT / "data/legal/clients.json", work / "clients.json")
        return [sys.executable, "-m", "sanitise_legal.legal", "build", "--data", str(work),
                "--store", str(store), "--out", str(work / "results")]  # fmt: skip
    return [sys.executable, "-m", "sanitise_finance", "build", "--input", str(corpus),
            "--store", str(store), "--no-index"]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=("legal", "finance"))
    ap.add_argument("--legal-docs", type=int, default=24)
    ap.add_argument("--finance-docs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "data", help="holds <domain>/store.sqlite")
    ap.add_argument("--mock", action="store_true", help="no model: rules and the canned rewrite")
    args = ap.parse_args()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    failed = []
    for domain in [args.only] if args.only else ["legal", "finance"]:
        store = args.out.resolve() / domain / "store.sqlite"
        store.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            docs = args.legal_docs if domain == "legal" else args.finance_docs
            cmd = command(domain, docs, Path(tmp), store) + (["--mock"] if args.mock else [])
            code = subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode
        print(f"{domain}: exit {code}, {store}", flush=True)
        if code or not store.exists():
            failed.append(domain)
    if failed:
        raise SystemExit(f"not built completely: {', '.join(failed)}. Run this again to resume.")


if __name__ == "__main__":
    main()
