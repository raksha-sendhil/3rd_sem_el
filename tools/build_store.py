"""Build the document store from Person D's corpus in small steps, with a checkpoint after each.

    python tools/build_store.py --docs 150 --checkpoint /content/drive/MyDrive/sanitise/store_150.sqlite

``sanitise build`` sanitises a whole corpus in one go and keeps its progress in a store on the
machine running it; on Colab that is lost when the runtime is reset. This runs it 25 documents at
a time and copies the store to ``--checkpoint`` (Google Drive) after every step. Run it again
after a disconnect, a reset or on another account with the same checkpoint file: it restores the
checkpoint, skips the documents already in it and goes on. At the end it builds the search index.

``--docs N`` (10-300) is ``tools/corpus_subset.py``'s subset of Person D's corpus: from 74 up,
every document the attacks and Target Info queries are about plus whole decoy people; below,
the 3 attack targets and their look-alikes first (a quick demo build). ``--docs 300`` is the
whole corpus; ``--docs fixtures`` the 20 hand-made test documents (no attack targets in them).
The local store holds one corpus at a time: building another size starts it afresh (each size
keeps its own checkpoint).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sanitise.rag.store import SqliteDocumentStore  # noqa: E402


def _in_store(path: Path) -> set[str]:
    if not path.exists():
        return set()
    store = SqliteDocumentStore(path)
    try:
        return set(store.ids())
    finally:
        store.close()


def _save(store: Path, checkpoint: Path) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = checkpoint.with_name(checkpoint.name + ".partial")
    shutil.copyfile(store, partial)
    partial.replace(checkpoint)  # never leave a half-copied checkpoint behind


def _sanitise(args: argparse.Namespace, corpus: Path, *extra: str) -> None:
    cmd = [sys.executable, "-m", "sanitise", "build", "--input", str(corpus)]
    cmd += ["--store", str(args.store), *extra] + (["--mock"] if args.mock else [])
    subprocess.run(cmd, cwd=ROOT, check=False)  # a failed document is retried on the next run


def _prune(store: Path, extra: set[str], index_dir: Path) -> None:
    """Drop documents that are no longer in the corpus (a checkpoint from before a corpus change);
    the search index is rebuilt from the store at the end."""
    if not extra:
        return
    db = sqlite3.connect(store)
    try:
        db.executemany("DELETE FROM documents WHERE doc_id = ?", [(d,) for d in extra])
        db.commit()
    finally:
        db.close()
    shutil.rmtree(index_dir, ignore_errors=True)
    print(f"removed {len(extra)} documents that are no longer in this corpus")


def _doc_id(record: dict) -> str:
    return hashlib.sha256(record["text"].encode()).hexdigest()[:16]


def main(args: argparse.Namespace) -> None:
    if args.docs == "fixtures":
        corpus = ROOT / "data" / "fixtures"
        records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(corpus.glob("*.json"))]
    else:
        subprocess.run(
            [sys.executable, "tools/corpus_subset.py", "--docs", args.docs], cwd=ROOT, check=True
        )
        corpus = ROOT / "data" / "corpus" / f"subset_{args.docs}.jsonl"
        records = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
    wanted = {_doc_id(r) for r in records}

    store = Path(args.store)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if args.fresh:  # after a pipeline change: every document is sanitised again
        for path in (store, checkpoint):
            if path and path.exists():
                path.unlink()
        shutil.rmtree(Path(args.index_dir), ignore_errors=True)
        print("starting fresh: the store and its checkpoint were removed")
    if _in_store(store) - wanted:  # another size was built here before: start this one afresh
        store.unlink()
        shutil.rmtree(Path(args.index_dir), ignore_errors=True)
        print("the local store held another corpus: starting afresh for this one")
    if checkpoint and checkpoint.exists() and len(_in_store(checkpoint)) > len(_in_store(store)):
        store.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint, store)
        print(f"restored {len(_in_store(store))} documents from {checkpoint}")
        _prune(store, _in_store(store) - wanted, Path(args.index_dir))

    done = _in_store(store)
    todo = [r for r in records if _doc_id(r) not in done]
    print(f"{len(records) - len(todo)} of {len(records)} documents already sanitised")

    with tempfile.TemporaryDirectory() as tmp:
        for start in range(0, len(todo), args.chunk):
            chunk = Path(tmp, f"chunk_{start}.jsonl")
            chunk.write_text(
                "".join(
                    json.dumps(r, ensure_ascii=False) + "\n" for r in todo[start:][: args.chunk]
                ),
                encoding="utf-8",
            )
            _sanitise(args, chunk, "--no-index")
            if checkpoint and store.exists():
                _save(store, checkpoint)
            print(
                f"== {len(_in_store(store))} of {len(records)} sanitised"
                + (", checkpoint saved" if checkpoint else ""),
                flush=True,
            )

    missing = len(wanted - _in_store(store))
    if missing:
        print(f"{missing} documents failed (often a daily limit): run this again to retry them")
        return
    _sanitise(args, corpus)  # every document is already in: this only builds the search index
    print(f"done: all {len(records)} documents sanitised and indexed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs", default="150", help="10-300, or fixtures")
    parser.add_argument("--store", default="data/store.sqlite")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument(
        "--fresh", action="store_true", help="remove the store and checkpoint, redo every document"
    )
    parser.add_argument("--checkpoint", help="copy of the store kept after every step (Drive)")
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--mock", action="store_true", help="no model (checks the plumbing)")
    args = parser.parse_args()
    if args.docs != "fixtures" and not (args.docs.isdigit() and 10 <= int(args.docs) <= 300):
        parser.error("--docs must be a number from 10 to 300, or fixtures")
    main(args)
