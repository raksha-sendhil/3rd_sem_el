"""Dashboard export and replay.  OWNER: Person C (with B for the demo).

Legal build spec, step 14: Kaggle does not accept incoming connections, so the dashboard runs
on a laptop and plays back a run's exported results. ``export_bundle`` copies what a run
produced into one folder:

    bundle/store.sqlite          originals and SanitisedRecords (the Documents view)
    bundle/clients.json          the client table (attack scoring, session monitor)
    bundle/attacks/*.json        the attack scripts
    bundle/recorded_calls.json   every answer-model call and retrieval of the run
    bundle/results.json, REPORT.md, charts/

With ``SANITISE_REPLAY_DIR=bundle`` the API serves that folder: retrieval returns the
recorded hits for a recorded question (BM25 over the stored originals otherwise), and the
answer model returns the recorded reply for a recorded prompt (a canned quote of the first
record otherwise). Because the store, the retrieval and the prompts are the same, every
question, comparison and attack of the run replays exactly, monitor verdicts included.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from sanitise.llm.mock import TEXT_CALL, MockCall, MockLLM, fabricate
from sanitise.mock_mode import canned_answer
from sanitise.monitor import build_monitor
from sanitise.rag.answer import RagContext
from sanitise.rag.retrieve import BM25Retriever
from sanitise.rag.store import SqliteDocumentStore
from sanitise.schemas import RetrievalHit


def _key(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()


def export_bundle(
    data_dir: str | Path, store_path: str | Path, results_dir: str | Path, bundle: str | Path
) -> Path:
    data, results, out = Path(data_dir), Path(results_dir), Path(bundle)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(store_path, out / "store.sqlite")
    shutil.copy2(data / "clients.json", out / "clients.json")
    shutil.copytree(data / "attacks", out / "attacks", dirs_exist_ok=True)
    for name in ("results.json", "recorded_calls.json", "REPORT.md"):
        if (results / name).is_file():
            shutil.copy2(results / name, out / name)
    if (results / "charts").is_dir():
        shutil.copytree(results / "charts", out / "charts", dirs_exist_ok=True)
    return out


class RecordedRetriever:
    def __init__(self, recorded: dict[str, list[dict[str, Any]]], fallback: BM25Retriever):
        self.recorded, self.fallback = recorded, fallback

    def search(self, query: str, k: int = 3) -> list[RetrievalHit]:
        hits = self.recorded.get(f"{k}\t{query}")
        if hits is None:
            return self.fallback.search(query, k)
        return [RetrievalHit.model_validate(h) for h in hits]


def replay_llm(bundle: str | Path) -> MockLLM:
    recorded = _recorded(bundle)
    replies = {_key(c["system"], c["user"]): c["text"] for c in recorded.get("llm", [])}

    def handler(call: MockCall) -> Any:
        if call.schema_name == TEXT_CALL:
            return replies.get(_key(call.system, call.user)) or canned_answer(call.user)
        return fabricate(call.json_schema)

    return MockLLM(handler=handler, model="replay")


def replay_context(bundle: str | Path, min_pool: int = 5) -> RagContext:
    root = Path(bundle)
    store = SqliteDocumentStore(root / "store.sqlite")
    fallback = BM25Retriever({d: store.original(d) for d in store.ids()})
    return RagContext(
        store=store,
        retriever=RecordedRetriever(_recorded(root).get("retrieval", {}), fallback),
        monitor=build_monitor(root / "clients.json", min_pool),
    )


def _recorded(bundle: str | Path) -> dict[str, Any]:
    path = Path(bundle) / "recorded_calls.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
