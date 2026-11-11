"""The legal build spec end to end: build the store, then run every evaluation.  OWNER: C.

``build_corpus_store`` is Part 2 steps 2-9 (Agents 1-3, leak check, storage) over
``data/legal/corpus``; it resumes, so a Kaggle restart loses at most one document.
``evaluate`` is steps 10-13: retrieval over the originals, both answer pipelines on the QA set,
every attack with and without the session monitor, the extraction and membership probes, and
the agent-level scores against the gold files. It writes ``results.json`` (every number the
report needs) and ``answers.jsonl`` (every answer), and records each model call and retrieval
so ``sanitise_legal.legal.export`` can replay the run in the dashboard.

    python -m sanitise_legal.legal build    --data data/legal --store data/legal/store.sqlite
    python -m sanitise_legal.legal evaluate --data data/legal --store data/legal/store.sqlite \\
        --out data/legal/results

``mock=True`` runs the same path without any model (rule pass, deterministic rewrite, the
canned answer model), which is how the pipeline is smoke-tested on a laptop.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from sanitise_legal.attacks.runner import load_attacks, load_people, run_attack
from sanitise_legal.build import BuildReport, build_store
from sanitise_legal.domain import use_domain
from sanitise_legal.evaluation import legal as metrics
from sanitise_legal.evaluation.facts import score_facts
from sanitise_legal.evaluation.rewrite import score_rewrite
from sanitise_legal.evaluation.scoring import load_dir, score
from sanitise_legal.llm.base import LLMClient
from sanitise_legal.loader import Document, sha256_hex
from sanitise_legal.monitor import AllowAllMonitor, build_monitor
from sanitise_legal.rag.answer import RagContext, compare
from sanitise_legal.rag.retrieve import Retriever
from sanitise_legal.rag.store import SqliteDocumentStore
from sanitise_legal.schemas import (
    ExtractorOutput,
    GoldRouting,
    QAItem,
    RagAnswer,
    RetrievalHit,
)

log = logging.getLogger(__name__)

DIRECT = ("NAME", "PHONE", "EMAIL", "ID_NUMBER", "CASE_NUMBER")


@dataclass
class RunConfig:
    data_dir: Path = Path("data/legal")
    store_path: Path = Path("data/legal/store.sqlite")
    out_dir: Path = Path("data/legal/results")
    k: int = 4
    retrieval_mode: str = "hybrid"
    index_dir: Path = Path("data/legal/index")
    embedding_model: str = "BAAI/bge-m3"
    min_pool: int = 5
    mi_limit: int = 100
    limit_docs: int | None = None

    def __post_init__(self) -> None:
        for name in ("data_dir", "store_path", "out_dir", "index_dir"):
            setattr(self, name, Path(getattr(self, name)))


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def corpus_documents(directory: Path, limit: int | None = None) -> list[Document]:
    docs = []
    for path in sorted(directory.glob("*.json"))[:limit]:
        row = _read(path)
        text = row["text"]
        docs.append(
            Document(doc_id=row["doc_id"], text=text, sha256=sha256_hex(text), source=path.name)
        )
    return docs


# ---------------------------------------------------------------- recording (for replay)


class RecordingLLM:
    """Wraps the answer model and keeps every (system, user) -> reply."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.model = getattr(inner, "model", type(inner).__name__)
        self.calls: list[dict[str, str]] = []

    async def json_call(self, system: str, user: str, schema: Any, retries: int | None = None):
        return await self.inner.json_call(system, user, schema, retries)

    async def text_call(self, system: str, user: str, **options: Any) -> str:
        text = await self.inner.text_call(system, user, **options)
        self.calls.append({"system": system, "user": user, "text": text})
        return text


class RecordingRetriever:
    def __init__(self, inner: Retriever) -> None:
        self.inner = inner
        self.calls: dict[str, list[dict[str, Any]]] = {}

    def search(self, query: str, k: int = 3) -> list[RetrievalHit]:
        hits = self.inner.search(query, k)
        self.calls[f"{k}\t{query}"] = [h.model_dump() for h in hits]
        return hits


# ---------------------------------------------------------------- build


async def build_corpus_store(
    cfg: RunConfig,
    llm: LLMClient,
    *,
    detector_llm: LLMClient | None = None,
    sanitiser: Callable[..., Awaitable[Any]] | None = None,
    progress: Callable[[int, Document, str], None] | None = None,
) -> BuildReport:
    """Agents 1-3 over the corpus into SQLite, plus one JSON file per document."""
    from sanitise_legal.agents.pipeline import sanitise

    with use_domain("legal"):
        run = sanitiser or partial(sanitise, detector_llm=detector_llm)
        return await build_store(
            corpus_documents(cfg.data_dir / "corpus", cfg.limit_docs),
            SqliteDocumentStore(cfg.store_path),
            llm,
            out_dir=cfg.out_dir.parent / "processed",
            sanitiser=run,
            progress=progress,
        )


# ---------------------------------------------------------------- evaluate


def _retriever(cfg: RunConfig, store: SqliteDocumentStore) -> Retriever:
    from sanitise_legal.rag.dense import build_retriever

    return build_retriever(
        store,
        mode=cfg.retrieval_mode,
        index_dir=cfg.index_dir,
        model_name=cfg.embedding_model,
    )


def _agent_scores(cfg: RunConfig, store: SqliteDocumentStore) -> dict[str, Any]:
    records = {d: store.sanitised(d) for d in store.ids()}
    gold = [g for g in load_dir(cfg.data_dir / "gold", gold=True) if g.doc_id in records]
    spans = score([records[g.doc_id].detector for g in gold], gold)
    direct = [spans.per_category.get(c, {}) for c in DIRECT]
    tp = sum(c.get("tp", 0) for c in direct)
    fn = sum(c.get("fn", 0) for c in direct)
    gold_facts = [
        ExtractorOutput.model_validate(_read(p))
        for p in sorted((cfg.data_dir / "gold_facts").glob("*.json"))
    ]
    gold_facts = [g for g in gold_facts if g.doc_id in records]
    gold_routing = [
        GoldRouting.model_validate(_read(p))
        for p in sorted((cfg.data_dir / "gold_routing").glob("*.json"))
    ]
    gold_routing = [g for g in gold_routing if g.doc_id in records]
    facts = score_facts([records[g.doc_id].extractor for g in gold_facts], gold_facts)
    rewrite = score_rewrite(
        [records[g.doc_id].rewriter for g in gold_routing], gold_routing, gold_facts
    )
    direct_leaks = sum(
        any(s.category.value in DIRECT for s in r.residual_leaks) for r in records.values()
    )
    return {
        "agent1": {
            "precision": spans.precision,
            "recall": spans.recall,
            "f1": spans.f1,
            "direct_recall": tp / (tp + fn) if tp + fn else 0.0,
            "per_category": spans.per_category,
            "documents": len(gold),
        },
        "agent2": facts.model_dump(),
        "agent3": {
            **rewrite.model_dump(),
            "documents_with_residual_leaks": sum(bool(r.residual_leaks) for r in records.values()),
            "documents_with_direct_leaks": direct_leaks,
            "fallbacks": dict(Counter(step for r in records.values() for step in r.meta.fallback)),
        },
        "offline_ms": {
            "agent1": [r.meta.detect_ms for r in records.values()],
            "agent2": [r.meta.extract_ms for r in records.values()],
            "agent3": [r.meta.rewrite_ms for r in records.values()],
        },
    }


@dataclass
class _Collected:
    answers: list[dict[str, Any]] = field(default_factory=list)

    def add(self, kind: str, item_id: str, a: RagAnswer, **extra: Any) -> None:
        self.answers.append({"kind": kind, "id": item_id, **extra, **a.model_dump()})


async def evaluate(
    cfg: RunConfig,
    answer_llm: LLMClient,
    *,
    retriever: Retriever | None = None,
    detect: metrics.Detector | None = None,
) -> dict[str, Any]:
    """Steps 10-13. ``detect`` is the rule pass used for the PII leakage rate."""
    with use_domain("legal"):
        return await _evaluate(cfg, answer_llm, retriever, detect)


async def _evaluate(
    cfg: RunConfig,
    answer_llm: LLMClient,
    retriever: Retriever | None,
    detect: metrics.Detector | None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    data = cfg.data_dir
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    store = SqliteDocumentStore(cfg.store_path)
    recorder = RecordingRetriever(retriever or _retriever(cfg, store))
    llm = RecordingLLM(answer_llm)
    clients = data / "clients.json"
    guarded = RagContext(
        store=store, retriever=recorder, monitor=build_monitor(clients, cfg.min_pool)
    )
    unguarded = RagContext(store=store, retriever=recorder, monitor=AllowAllMonitor())
    collected = _Collected()
    indexed = set(store.ids())

    # QA set: both pipelines, a fresh session per question
    qa = [QAItem.model_validate(q) for q in _read(data / "qa.json")]
    qa = [q for q in qa if set(q.doc_ids) & indexed]
    qa_answers: list[RagAnswer] = []
    for q in qa:
        res = await compare(guarded, q.question, guarded.sessions.create(), llm, cfg.k)
        for a in (res.baseline, res.protected):
            qa_answers.append(a)
            collected.add("qa", q.qa_id, a)
    ranked = {q.question: [h.doc_id for h in recorder.search(q.question, 8)] for q in qa}

    # attacks, with the monitor and without it (spec: "without the monitor: later than
    # baseline; with the monitor: never")
    people = load_people(clients)
    attack_runs: dict[str, Any] = {}
    attack_answers: list[RagAnswer] = []
    for attack in load_attacks(data / "attacks").values():
        for label, ctx in (("monitor", guarded), ("no_monitor", unguarded)):
            run = await run_attack(ctx, attack, people, llm)
            attack_runs[f"{attack.attack_id}:{label}"] = run.model_dump()
            for r in run.rounds:
                for a in (r.baseline, r.protected):
                    if label == "monitor":
                        attack_answers.append(a)
                    collected.add("attack", attack.attack_id, a, monitor=label, round=r.round)

    # extraction probes (metric 1)
    by_doc = {d.doc_id: d for d in corpus_documents(data / "corpus")}
    client_rows = {c["client_id"]: c for c in _read(clients)}
    extraction_rows: dict[str, list[tuple[str, str, list[str]]]] = {"baseline": [], "protected": []}
    extraction_answers: list[RagAnswer] = []
    for probe in _read(data / "extraction.json"):
        if probe["doc_id"] not in indexed:
            continue
        c = client_rows[probe["client_id"]]
        ids = [c["name"], c["phone"], c["email"], c["pan"], c["case_number"]]
        res = await compare(guarded, probe["query"], guarded.sessions.create(), llm, cfg.k)
        for a in (res.baseline, res.protected):
            extraction_rows[a.pipeline].append((a.answer, by_doc[probe["doc_id"]].text, ids))
            extraction_answers.append(a)
            collected.add("extraction", probe["prompt_id"], a)

    # membership inference (metric 3): the answer's similarity to the probed document
    nonmember_text = {d.doc_id: d.text for d in corpus_documents(data / "nonmembers" / "docs")}
    probes = _read(data / "membership.json")
    mi: dict[str, dict[str, list[float]]] = {
        p: {"members": [], "nonmembers": []} for p in ("baseline", "protected")
    }
    for group in ("members", "nonmembers"):
        for probe in probes[group][: cfg.mi_limit]:
            doc = by_doc.get(probe["doc_id"])
            text = doc.text if doc else nonmember_text.get(probe["doc_id"], "")
            if group == "members" and probe["doc_id"] not in indexed:
                continue
            res = await compare(unguarded, probe["query"], unguarded.sessions.create(), llm, cfg.k)
            for a in (res.baseline, res.protected):
                mi[a.pipeline][group].append(metrics.rouge_l(a.answer, text))
                collected.add("membership", probe["doc_id"], a, group=group)

    # PII leakage over every answer (QA, attacks, extraction)
    if detect is None:
        from sanitise_legal.rules import rule_detect as detect
    every = qa_answers + attack_answers + extraction_answers
    agent = _agent_scores(cfg, store)
    results: dict[str, Any] = {
        "config": {k: str(v) for k, v in cfg.__dict__.items()},
        "answer_model": llm.model,
        "documents_indexed": len(indexed),
        "pipelines": {},
        "recall_at_k": metrics.recall_at_k(ranked, qa),
        "attacks": attack_runs,
        "agents": {k: v for k, v in agent.items() if k != "offline_ms"},
        "epsilon_delta": "none (no differential-privacy noise is added)",
    }
    for p in ("baseline", "protected"):
        results["pipelines"][p] = {
            "extraction": metrics.extraction_leakage(extraction_rows[p]),
            "pii_leakage": metrics.pii_leakage(
                [a.answer for a in every if a.pipeline == p], detect
            ),
            "membership": metrics.membership_inference(mi[p]["members"], mi[p]["nonmembers"]),
        }
    results["answer_quality"] = metrics.answer_quality(qa_answers, qa)
    results["latency"] = metrics.latency(qa_answers, agent["offline_ms"])
    results["attack_outcomes"] = attack_outcomes(attack_runs)
    results["seconds"] = time.perf_counter() - t0

    out = cfg.out_dir
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out / "answers.jsonl").open("w", encoding="utf-8") as f:
        for row in collected.answers:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "recorded_calls.json").write_text(
        json.dumps({"llm": llm.calls, "retrieval": recorder.calls}, ensure_ascii=False),
        encoding="utf-8",
    )
    return results


def attack_outcomes(runs: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per attack and monitor setting: when each pipeline identified the target and
    the smallest candidate pool the protected answers left."""
    rows = []
    for key, run in sorted(runs.items()):
        attack_id, label = key.split(":")
        pools = [
            r["protected_leakage"]["candidate_pool"]
            for r in run["rounds"]
            if r.get("protected_leakage")
        ]
        blocked = [
            r["round"] for r in run["rounds"] if r["protected"]["monitor"]["verdict"] == "blocked"
        ]
        rows.append(
            {
                "attack_id": attack_id,
                "kind": attack_id.split("_")[0],
                "monitor": label,
                "rounds": len(run["rounds"]),
                "baseline_identified_at": run["baseline_identified_at"],
                "protected_identified_at": run["protected_identified_at"],
                "protected_min_pool": min(pools) if pools else None,
                "blocked_rounds": blocked,
            }
        )
    return rows
