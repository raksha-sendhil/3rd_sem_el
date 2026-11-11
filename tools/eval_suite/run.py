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
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_REPO = HERE.parents[1]

SCALES = {
    "full": {"indexed": 500, "heldout": 250, "targeted": 250, "repeat": 250, "members": 500,
             "nonmembers": 250, "utility": 250, "targets": 60, "e2e": 50, "gen_docs": None},
    "quick": {"indexed": 30, "heldout": 15, "targeted": 10, "repeat": 10, "members": 10,
              "nonmembers": 10, "utility": 10, "targets": 2, "e2e": 3, "gen_docs": 30},
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
    p.add_argument("--scale", choices=list(SCALES), default="full")
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
    if D.name == "health":
        targeted = _load(ARGS.repo / "data/attacks/target_info_queries.json")
        targeted = [t for t in targeted][: SCALE["targeted"] if ARGS.scale == "quick" else None]
    else:
        targeted = Q.targeted_queries(D, people, SCALE["targeted"])
    _dump(data / "targeted.json", targeted)
    _dump(data / "sessions.json", sessions)

    # the generated corpus (a quick run keeps the people asked about plus a few others)
    gen = data / "gen" / "corpus"
    gen.mkdir(parents=True, exist_ok=True)
    files = sorted(D.corpus_dir.glob("*.json"))
    if SCALE["gen_docs"]:
        wanted_people = {s["target"] for s in sessions} | {t["person_id"] for t in targeted}
        wanted = {d for p in people if p.person_id in wanted_people for d in p.doc_ids}
        keep = [f for f in files if _load(f)["doc_id"] in wanted]
        keep += [f for f in files if f not in keep][: max(0, SCALE["gen_docs"] - len(keep))]
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
    from sanitise.rag.answer import RagContext
    from sanitise.rag.dense import build_retriever
    from sanitise.rag.store import SqliteDocumentStore

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


async def _ask(ctx, llm, items: list[dict], path: Path) -> None:
    """Each item once per pipeline, in a fresh session; one line per item in ``path``."""
    from sanitise.rag.answer import answer

    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(path)
    todo = [it for it in items if it["id"] not in done]
    print(f"{path.stem}: {len(todo)} of {len(items)} questions to ask", flush=True)
    with path.open("a", encoding="utf-8") as f:
        for i, it in enumerate(todo, 1):
            session = ctx.sessions.create()
            row = dict(it)
            try:
                for pipeline in PIPELINES:
                    row[pipeline] = _slim(
                        await answer(ctx, it["query"], pipeline, session, llm, D.k)
                    )
            except Exception as exc:  # a model error: skipped now, asked again on the next run
                print(f"  {it['id']}: {str(exc)[:160]}", flush=True)
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {path.stem}: {i}/{len(todo)}", flush=True)


async def _with_llm(work) -> None:
    from sanitise.llm.factory import close_llm_client

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
    from sanitise.rag.answer import answer

    sessions = _load(WORK / "data" / "sessions.json")
    path = WORK / "answers" / "sessions.jsonl"

    async def work(llm) -> None:
        ctx = _context("gen")
        done = _done(path)
        todo = [s for s in sessions if s["id"] not in done]
        print(f"sessions: {len(todo)} of {len(sessions)} to play ({Q.SESSION_ROUNDS} rounds each)")
        with path.open("a", encoding="utf-8") as f:
            for i, s in enumerate(todo, 1):
                session = ctx.sessions.create()
                rounds = []
                try:
                    for t, query in enumerate(s["queries"], 1):
                        r = {"round": t, "query": query}
                        for pipeline in PIPELINES:
                            a = await answer(ctx, query, pipeline, session, llm, D.k)
                            r[pipeline] = _slim(a)
                        rounds.append(r)
                except Exception as exc:
                    print(f"  {s['id']}: {str(exc)[:160]} (played again on the next run)")
                    continue
                f.write(json.dumps({**s, "rounds": rounds}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{i}/{len(todo)}] {s['id']}", flush=True)

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

        e2e.prepare(D.name, ARGS.main_repo, WORK, SCALE["e2e"])
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
