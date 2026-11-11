"""Run the evaluation of docs/evaluation_metrics.md for one domain, step by step.

    python tools/eval_suite/run.py data   --domain health --work /kaggle/working/eval/health
    python tools/eval_suite/run.py build  --domain health --work ...
    python tools/eval_suite/run.py single --domain health --work ...
    python tools/eval_suite/run.py multi  --domain health --work ...
    python tools/eval_suite/run.py e2e-prep / e2e-label / e2e --domain health --work ...
    python tools/eval_suite/run.py report --domain health --work ...

Every step resumes: what is already on disk under ``--work`` is skipped, so a step stopped by a
Kaggle time limit goes on where it stopped when run again.

``--repo`` is the clone whose ``sanitise`` package does the work (main for health, the legal
branch for legal); this suite always comes from main. ``e2e-prep`` needs main's OCR, so for
legal it runs with ``--repo`` pointing at the main clone. ``--scale quick`` runs every step on
a few items, to check the set-up before the full run.

Configurations: **no privacy** (standard RAG over the original documents) and **ours** (Agents
1-3's rewrites, with the session monitor), on the same questions, retrieval and answer model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_REPO = HERE.parents[1]

# "full": the plan's volumes (about 12-15 GPU hours per domain); "5h": the same metrics on less
# data, about 4.5 hours per domain; "quick": a few items per step (about 10 minutes) to check
# the set-up. gen_docs: size of the generated store (None = all 300); e2e_parts: parts kept per
# end-to-end document.
SCALES = {
    "full": {"indexed": 500, "heldout": 250, "targeted": 250, "repeat": 250, "members": 500,
             "nonmembers": 250, "utility": 250, "targets": 60, "e2e": 50, "gen_docs": None,
             "e2e_parts": 2, "asked_people": None},
    "5h": {"indexed": 150, "heldout": 100, "targeted": 100, "repeat": 100, "members": 100,
           "nonmembers": 100, "utility": 100, "targets": 20, "e2e": 25, "gen_docs": 150,
           "e2e_parts": 1, "asked_people": 35},
    "quick": {"indexed": 12, "heldout": 6, "targeted": 5, "repeat": 5, "members": 5,
              "nonmembers": 5, "utility": 5, "targets": 1, "e2e": 2, "gen_docs": 15,
              "e2e_parts": 1, "asked_people": 3},
}  # fmt: skip
PIPELINES = ("baseline", "protected")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["data", "build", "single", "multi", "e2e-prep", "e2e-label",
                                    "e2e", "report"])  # fmt: skip
    p.add_argument("--domain", required=True, choices=["health", "legal"])
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--repo", type=Path, default=MAIN_REPO)
    p.add_argument("--main-repo", type=Path, default=MAIN_REPO)
    p.add_argument("--scale", choices=list(SCALES), default="5h")
    p.add_argument("--stores", default="gen,real,e2e_gold,e2e_ocr")
    return p.parse_args()


ARGS = _args()
os.chdir(ARGS.repo)  # relative paths in the branch's settings (ontology, config/domains.json)
sys.path.insert(0, str(ARGS.repo / "src"))
sys.path.insert(0, str(HERE.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WORK: Path = ARGS.work.resolve()
WORK.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SANITISE_MONITOR_DB_PATH", str(WORK / "stores" / "monitor.sqlite"))

from eval_suite import questions as Q  # noqa: E402
from eval_suite.domains import get_domain  # noqa: E402

SCALE = SCALES[ARGS.scale]
D = get_domain(ARGS.domain, ARGS.repo)


def _dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- step: data


def step_data() -> None:
    from eval_suite import realdata

    data = WORK / "data"
    people = D.people()
    if not people:
        raise SystemExit(f"no person table at {D.people_path}")

    # questions on the generated corpus: Target Info and the attack sessions
    sessions = Q.attack_sessions(D, people, SCALE["targets"])
    targets = [s["target"] for s in sessions]
    if D.name == "health":  # Person D's fixed set, spread evenly over its 29 people
        targeted = _load(ARGS.repo / "data/attacks/target_info_queries.json")
        n = min(SCALE["targeted"], len(targeted))
        targeted = [targeted[int(i * len(targeted) / n)] for i in range(n)]
    else:  # about the session targets first, then others, a few questions each
        asked = list(dict.fromkeys(targets))
        rest = [
            p.person_id
            for p in sorted(people, key=lambda p: p.person_id)
            if p.person_id not in asked
        ]
        cap = SCALE["asked_people"] or len(people)
        asked = set((asked + rest)[:cap])
        targeted = Q.targeted_queries(
            D, [p for p in people if p.person_id in asked], SCALE["targeted"]
        )
    _dump(data / "targeted.json", targeted)
    _dump(data / "sessions.json", sessions)

    # the generated corpus: every document of the people asked about, then whole other people
    # (decoys, so candidate pools stay realistic) up to gen_docs
    gen = data / "gen" / "corpus"
    gen.mkdir(parents=True, exist_ok=True)
    by_id = {_load(f)["doc_id"]: f for f in sorted(D.corpus_dir.glob("*.json"))}
    files = list(by_id.values())
    if SCALE["gen_docs"]:
        wanted = set(targets) | {t["person_id"] for t in targeted}
        order = [p for p in people if p.person_id in wanted]
        others = [p for p in people if p.person_id not in wanted]
        random.Random("decoys").shuffle(others)
        keep: list[Path] = []
        for p in order + others:
            if p.person_id not in wanted and len(keep) >= SCALE["gen_docs"]:
                break
            keep += [by_id[d] for d in p.doc_ids if d in by_id]
        files = keep
    for f in files:
        shutil.copyfile(f, gen / f.name)
    print(
        f"generated corpus: {len(files)} documents, {len(people)} people; "
        f"{len(targeted)} Target Info questions; {len(sessions)} attack sessions"
    )

    # real public text
    fetch = realdata.chatdoctor if D.name == "health" else realdata.cuad
    real = fetch(SCALE["indexed"], SCALE["heldout"], WORK / ".cache")
    corpus = data / "real" / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for d in real["indexed"]:
        _dump(corpus / f"{d['doc_id']}.json", d)
    _dump(data / "real" / "heldout.json", real["heldout"])
    _dump(data / "utility.json", real["utility"][: SCALE["utility"]])
    _dump(data / "repeat.json", Q.repeat_prompts(real["heldout"], SCALE["repeat"]))
    _dump(data / "membership.json", Q.membership_probes(real["indexed"], real["heldout"],
                                                        SCALE["members"], SCALE["nonmembers"]))  # fmt: skip
    print(
        f"real corpus: {len(real['indexed'])} indexed, {len(real['heldout'])} held out; "
        f"{len(real['utility'][: SCALE['utility']])} utility questions"
    )


# ---------------------------------------------------------------- step: build

STORE_DATA = {"gen": "data/gen", "real": "data/real", "e2e_gold": "e2e/gold", "e2e_ocr": "e2e/ocr"}


def step_build() -> None:
    for name in ARGS.stores.split(","):
        data_dir = WORK / STORE_DATA[name]
        if not (data_dir / "corpus").is_dir():
            print(f"{name}: no documents at {data_dir / 'corpus'} yet, skipped")
            continue
        n = len(list((data_dir / "corpus").glob("*.json")))
        print(f"{name}: Agents 1-3 over {n} documents (resumes)", flush=True)
        store = WORK / "stores" / f"{name}.sqlite"
        store.parent.mkdir(parents=True, exist_ok=True)
        t = time.perf_counter()
        D.build(data_dir, store)
        print(f"{name}: done in {(time.perf_counter() - t) / 60:.0f} min", flush=True)


# ---------------------------------------------------------------- answering


def _context(name: str):
    RagContext = D.mod("rag.answer").RagContext
    build_retriever = D.mod("rag.dense").build_retriever
    SqliteDocumentStore = D.mod("rag.store").SqliteDocumentStore

    path = WORK / "stores" / f"{name}.sqlite"
    if not path.exists():
        raise SystemExit(f"no store {path}: run the build step first")
    store = SqliteDocumentStore(path)
    retriever = build_retriever(
        store, mode=os.environ.get("SANITISE_RETRIEVAL_MODE", "hybrid"),
        index_dir=WORK / "stores" / f"index_{name}", model_name="BAAI/bge-m3",
    )  # fmt: skip
    return RagContext(store=store, retriever=retriever, monitor=D.monitor(store))


def _slim(a) -> dict:
    return {"answer": a.answer, "doc_ids": a.doc_ids, "latency_ms": a.latency_ms,
            "verdict": a.monitor.verdict}  # fmt: skip


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(x)["id"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}


# Items answered at once. Within an item both pipelines are asked together (the baseline has
# no monitor, so the order does not matter), so up to 2 x CONCURRENCY requests reach the model;
# start Ollama with OLLAMA_NUM_PARALLEL at least that (the notebooks do).
CONCURRENCY = max(1, int(os.environ.get("EVAL_CONCURRENCY", "2")))


async def _pair(ctx, llm, query: str, session: str) -> dict:
    answer = D.mod("rag.answer").answer

    base, prot = await asyncio.gather(
        answer(ctx, query, "baseline", session, llm, D.k),
        answer(ctx, query, "protected", session, llm, D.k),
    )
    return {"baseline": _slim(base), "protected": _slim(prot)}


async def _run_all(todo: list[dict], play, path: Path, label: str) -> None:
    """``play(item)`` for every item, CONCURRENCY at a time; each finished item is one line."""
    sem = asyncio.Semaphore(CONCURRENCY)
    finished = 0
    with path.open("a", encoding="utf-8") as f:

        async def one(item: dict) -> None:
            nonlocal finished
            async with sem:
                try:
                    row = await play(item)
                except Exception as exc:  # a model error: skipped now, asked again next run
                    print(f"  {item['id']}: {str(exc)[:160]} (again on the next run)", flush=True)
                    return
            f.write(json.dumps(row, ensure_ascii=False) + chr(10))
            f.flush()
            finished += 1
            if finished % 10 == 0 or finished == len(todo):
                print(f"  {label}: {finished}/{len(todo)}", flush=True)

        await asyncio.gather(*(one(it) for it in todo))


async def _ask(ctx, llm, items: list[dict], path: Path) -> None:
    """Each item once per pipeline, in a fresh session; one line per item in ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(path)
    todo = [it for it in items if it["id"] not in done]
    print(f"{path.stem}: {len(todo)} of {len(items)} questions to ask", flush=True)

    async def play(it: dict) -> dict:
        return {**it, **await _pair(ctx, llm, it["query"], ctx.sessions.create())}

    await _run_all(todo, play, path, path.stem)


async def _with_llm(work) -> None:
    close_llm_client = D.mod("llm.factory").close_llm_client

    llm, label = D.answer_llm()
    _dump(WORK / "answers" / "answer_model.json", {"answer_model": label, "k": D.k})
    try:
        with D.context():
            await work(llm)
    finally:
        await close_llm_client(llm)


def step_single() -> None:
    data = WORK / "data"
    targeted = [{"id": t["query_id"], **t} for t in _load(data / "targeted.json")]

    async def work(llm) -> None:
        await _ask(_context("gen"), llm, targeted, WORK / "answers" / "target_info.jsonl")
        real = _context("real")
        for name in ("repeat", "membership", "utility"):
            await _ask(real, llm, _load(data / f"{name}.json"), WORK / "answers" / f"{name}.jsonl")

    asyncio.run(_with_llm(work))


def step_multi() -> None:
    sessions = _load(WORK / "data" / "sessions.json")
    path = WORK / "answers" / "sessions.jsonl"

    async def work(llm) -> None:
        ctx = _context("gen")
        done = _done(path)
        todo = [s for s in sessions if s["id"] not in done]
        print(f"sessions: {len(todo)} of {len(sessions)} to play ({Q.SESSION_ROUNDS} rounds each)")

        async def play(s: dict) -> dict:
            session = ctx.sessions.create()  # one session per attack, its rounds in order
            rounds = []
            for t, query in enumerate(s["queries"], 1):
                rounds.append({"round": t, "query": query, **await _pair(ctx, llm, query, session)})
            return {**s, "rounds": rounds}

        path.parent.mkdir(parents=True, exist_ok=True)
        await _run_all(todo, play, path, "sessions")

    asyncio.run(_with_llm(work))


def step_e2e() -> None:
    from eval_suite import e2e

    rows = e2e.questions(WORK)

    async def work(llm) -> None:
        for side in ("gold", "ocr"):
            await _ask(_context(f"e2e_{side}"), llm, rows, WORK / "answers" / f"e2e_{side}.jsonl")

    asyncio.run(_with_llm(work))


def main() -> None:
    step = ARGS.step
    if step == "data":
        step_data()
    elif step == "build":
        step_build()
    elif step == "single":
        step_single()
    elif step == "multi":
        step_multi()
    elif step == "e2e-prep":
        from eval_suite import e2e

        e2e.prepare(D.name, ARGS.main_repo, WORK, SCALE["e2e"], SCALE["e2e_parts"])
    elif step == "e2e-label":
        from eval_suite import e2e

        e2e.label(WORK)
    elif step == "e2e":
        step_e2e()
    elif step == "report":
        from eval_suite import report

        report.write(D, WORK, ARGS.scale)


if __name__ == "__main__":
    main()
