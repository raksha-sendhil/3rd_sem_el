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

# "high": the clean test's volumes (about 8 hours per domain); "5h" and "2h": the same metrics on
# less data; "full": more than one Kaggle session holds; "quick": a few items per step (about 10
# minutes) to check the set-up, on data development has already seen. Every scale but "quick" is
# the clean test: unseen real data (seen.json) and new people (population.py). gen_docs: size of the generated store (None = all 300); e2e_parts: parts kept per
# end-to-end document.
SCALES = {
    "full": {"indexed": 500, "heldout": 250, "targeted": 250, "repeat": 250, "members": 500,
             "nonmembers": 250, "utility": 250, "targets": 60, "e2e": 50, "gen_docs": None,
             "e2e_parts": 2, "asked_people": None, "person_members": 30, "pii_docs": 200},
    "high": {"indexed": 300, "heldout": 150, "targeted": 200, "repeat": 200, "members": 200,
             "nonmembers": 150, "utility": 200, "targets": 40, "e2e": 50, "gen_docs": 300,
             "e2e_parts": 1, "asked_people": 70, "person_members": 30, "pii_docs": 200},
    "5h": {"indexed": 150, "heldout": 100, "targeted": 100, "repeat": 100, "members": 100,
           "nonmembers": 100, "utility": 100, "targets": 20, "e2e": 25, "gen_docs": 150,
           "e2e_parts": 1, "asked_people": 35, "person_members": 20, "pii_docs": 100},
    "2h": {"indexed": 100, "heldout": 60, "targeted": 60, "repeat": 60, "members": 60,
           "nonmembers": 60, "utility": 60, "targets": 10, "e2e": 20, "gen_docs": 100,
           "e2e_parts": 1, "asked_people": 20, "person_members": 10, "pii_docs": 50},
    "night": {"indexed": 60, "heldout": 40, "targeted": 40, "repeat": 40, "members": 40,
              "nonmembers": 40, "utility": 40, "targets": 8, "e2e": 10, "gen_docs": 60,
              "e2e_parts": 1, "asked_people": 15, "person_members": 10, "pii_docs": 30},
    "quick": {"indexed": 12, "heldout": 6, "targeted": 5, "repeat": 5, "members": 5,
              "nonmembers": 5, "utility": 5, "targets": 1, "e2e": 2, "gen_docs": 15,
              "e2e_parts": 1, "asked_people": 3, "person_members": 3, "pii_docs": 4},
}  # fmt: skip
PIPELINES = ("baseline", "protected")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["data", "build", "single", "multi", "e2e-prep", "e2e-label",
                                    "utility-questions", "e2e", "report"])  # fmt: skip
    p.add_argument("--domain", required=True, choices=["health", "legal", "finance"])
    p.add_argument("--source", default="", help="finance: bank, kyc, corporate or cheques")
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--repo", type=Path, default=MAIN_REPO)
    p.add_argument("--main-repo", type=Path, default=MAIN_REPO)
    p.add_argument("--scale", choices=list(SCALES), default="high")
    p.add_argument("--people-file", type=Path, help="the clean test's people (population.py)")
    p.add_argument("--stores", default="gen,real,e2e_gold,e2e_ocr,pii")
    p.add_argument("--only", default="target_info,repeat,membership,person_membership,utility",
                   help="single: which question sets to ask")  # fmt: skip
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
D = get_domain(ARGS.domain, ARGS.repo, ARGS.source)
CLEAN = ARGS.scale != "quick" and D.name != "finance"  # finance: each source's own external data
if CLEAN:  # the clean population replaces the development corpus and its person table
    D.people_path = WORK / "data" / "population" / "people.json"
    D.corpus_dir = WORK / "data" / "population" / "docs"


# ---------------------------------------------------------------- time limit
# EVAL_DEADLINE (a Unix time, set by the notebook) ends the run at a fixed time: the build stops
# early enough to leave ANSWER_RESERVE_S for the questions and the report; question steps skip
# what they cannot ask before REPORT_RESERVE_S; the report says what was built and skipped.
ANSWER_RESERVE_S = 75 * 60  # the smallest scales; _answer_reserve() grows it with the questions
REPORT_RESERVE_S = 10 * 60


def _answer_reserve() -> float:
    """Time kept back from the build for the questions, sessions, scans and the report, from the
    speeds of earlier runs on a T4 (about 2.5 s per single question with two at a time, 1 s per
    session turn, 2 s per scan question), with half again as margin; at least 75 minutes."""
    single = SCALE["targeted"] + SCALE["repeat"] + SCALE["members"] + SCALE["nonmembers"] +         SCALE["utility"] + 4 * SCALE.get("person_members", 0)
    turns = SCALE["targets"] * 3 * 8
    scans = SCALE["e2e"] * 6
    return max(ANSWER_RESERVE_S, 1.5 * (2.5 * single + 1.0 * turns + 2.0 * scans) + 600)


def _left() -> float:
    """Seconds until EVAL_DEADLINE; infinity without one."""
    deadline = os.environ.get("EVAL_DEADLINE")
    return float(deadline) - time.time() if deadline else float("inf")


def _note_skipped(label: str, asked: int, skipped: int, failed: int = 0) -> None:
    path = WORK / "answers" / "skipped.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old[label] = {"to_ask": asked, "skipped": skipped, **({"failed": failed} if failed else {})}
    _dump(path, old)


def _fetch_embedder(tries: int = 6) -> None:
    """Download the retrieval embedding model now, with retries: it is first needed by the
    question steps hours later, when a HuggingFace failure would leave every question unasked.
    A failure here stops the data step, early and cheaply."""
    if os.environ.get("EVAL_MOCK") or os.environ.get("SANITISE_RETRIEVAL_MODE", "hybrid") != "hybrid":
        return
    for attempt in range(tries):
        try:
            from huggingface_hub import snapshot_download

            snapshot_download("BAAI/bge-m3")
            print("embedding model BAAI/bge-m3 ready", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            wait = min(300, 20 * 2**attempt)
            print(f"embedding model download failed ({exc}); again in {wait} s", flush=True)
            time.sleep(wait)
    raise SystemExit("could not download the embedding model BAAI/bge-m3 from HuggingFace")


def _gemini_budget(most_s: float, share: float) -> None:
    """A step that writes questions with Gemini (labels, utility questions) stops asking after
    ``most_s`` or ``share`` of the time left, whichever is less, so the build keeps its time
    even when every model is at its limit. No limit on an untimed run."""
    if _left() == float("inf"):
        os.environ.pop("EVAL_GEMINI_STOP_AT", None)
        return
    span = max(0.0, min(most_s, (_left() - _answer_reserve()) * share))
    os.environ["EVAL_GEMINI_STOP_AT"] = str(time.time() + span)
    print(f"Gemini question writing: at most {span / 60:.0f} min for this step", flush=True)


def _dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- step: data


def step_data() -> None:
    from eval_suite import realdata

    data = WORK / "data"
    made: dict = {}
    if CLEAN:
        from eval_suite import population

        file = ARGS.people_file or ARGS.repo / population.PEOPLE_FILE
        made = population.build(D.name, file, data / "population", ARGS.repo)
    people = D.people() if D.people_path.exists() else []
    if not people:
        raise SystemExit(f"no person table at {D.people_path}")

    # questions on the generated corpus: Target Info and the attack sessions
    sessions = Q.attack_sessions(D, people, SCALE["targets"])
    targets = [s["target"] for s in sessions]
    if D.name == "health" and not CLEAN:  # Person D's fixed set, spread evenly over its 29 people
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
    wanted = set(targets) | {t["person_id"] for t in targeted}
    # person-level membership: people never asked about whose records are kept out of the store
    spare = sorted((p for p in people if p.person_id not in wanted), key=lambda p: p.person_id)
    random.Random("person-membership").shuffle(spare)
    outsiders = spare[: SCALE.get("person_members", 0)]
    out_ids = {p.person_id for p in outsiders}
    left_out = {d for p in outsiders for d in p.doc_ids}
    files = [f for i, f in by_id.items() if i not in left_out]
    if SCALE["gen_docs"]:
        order = [p for p in people if p.person_id in wanted]
        others = [p for p in people if p.person_id not in wanted and p.person_id not in out_ids]
        random.Random("decoys").shuffle(others)
        keep: list[Path] = []
        for p in order + others:
            if p.person_id not in wanted and len(keep) >= SCALE["gen_docs"]:
                break
            keep += [by_id[d] for d in p.doc_ids if d in by_id]
        files = keep
    for f in files:
        shutil.copyfile(f, gen / f.name)
    stored = {_load(f)["doc_id"] for f in files}
    insiders = [p for p in people if p.doc_ids and set(p.doc_ids) <= stored]
    random.Random("person-membership-members").shuffle(insiders)
    _dump(data / "person_membership.json",
          Q.person_membership_probes(insiders[: len(outsiders)], outsiders))  # fmt: skip
    print(
        f"generated corpus: {len(files)} documents, {len(people)} people; "
        f"{len(targeted)} Target Info questions; {len(sessions)} attack sessions"
    )

    # real public text
    fetch = realdata.chatdoctor if D.name == "health" else realdata.cuad
    options = {}
    if CLEAN:
        options = {"seed": realdata.CLEAN_SEED, "exclude": realdata.seen(D.name)}
    real = fetch(SCALE["indexed"], SCALE["heldout"], WORK / ".cache", **options)
    corpus = data / "real" / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for d in real["indexed"]:
        _dump(corpus / f"{d['doc_id']}.json", d)
    _dump(data / "real" / "heldout.json", real["heldout"])
    _dump(data / "utility.json", real["utility"][: SCALE["utility"]])
    _dump(data / "repeat.json", Q.repeat_prompts(real["heldout"], SCALE["repeat"]))
    _dump(data / "membership.json", Q.membership_probes(real["indexed"], real["heldout"],
                                                        SCALE["members"], SCALE["nonmembers"]))  # fmt: skip
    _dump(data / "provenance.json", {
        "clean": CLEAN, **made, "generated_documents_indexed": len(files),
        "attack_sessions": len(sessions), "target_info_questions": len(targeted),
        "person_membership_each_group": len(outsiders),
        "real_indexed": len(real["indexed"]), "real_heldout": len(real["heldout"]),
        "utility_questions": len(real["utility"][: SCALE["utility"]]),
        "seed": options.get("seed", 2026), "excluded_as_seen": len(options.get("exclude", ())),
    })  # fmt: skip
    print(
        f"real corpus: {len(real['indexed'])} indexed, {len(real['heldout'])} held out; "
        f"{len(real['utility'][: SCALE['utility']])} utility questions"
    )
    from eval_suite import residual  # unseen annotated documents: what the agents leave

    residual.fetch_pii(D.name, SCALE.get("pii_docs", 0), WORK / "pii", WORK / ".cache")


# ---------------------------------------------------------------- step: build

STORE_DATA = {"gen": "data/gen", "real": "data/real", "e2e_gold": "e2e/gold", "e2e_ocr": "e2e/ocr",
              "pii": "pii"}


def step_build() -> None:
    todo: dict[str, tuple[Path, Path, int]] = {}
    for name in ARGS.stores.split(","):
        data_dir = WORK / STORE_DATA[name]
        if not (data_dir / "corpus").is_dir():
            print(f"{name}: no documents at {data_dir / 'corpus'} yet, skipped")
            continue
        n = len(list((data_dir / "corpus").glob("*.json")))
        store = WORK / "stores" / f"{name}.sqlite"
        store.parent.mkdir(parents=True, exist_ok=True)
        todo[name] = (data_dir, store, n)
    # a document that failed (Gemini busy for longer than its retries) is tried again in later
    # passes, after growing pauses, while a store is under GOOD_ENOUGH built; retrying stops when
    # two passes in a row add nothing (Gemini down: waiting longer will not help) or time runs
    # short, and the run then scores what was built
    pauses = QUICK_RETRY_PAUSES_S if ARGS.scale == "quick" else BUILD_RETRY_PAUSES_S
    stalled, before = 0, None
    for attempt, pause in enumerate([0, *pauses]):
        first = attempt == 0
        missing = {k: v for k, v in todo.items()
                   if _built(v[1]) < v[2] and (first or _built(v[1]) < GOOD_ENOUGH * v[2])}  # fmt: skip
        if not missing:
            break
        now = sum(_built(v[1]) for v in todo.values())
        if before is not None:
            stalled = stalled + 1 if now == before else 0
            if stalled >= 2:
                print("build: two retry passes in a row added nothing; scoring what was built",
                      flush=True)  # fmt: skip
                break
        before = now
        if pause and _left() - _answer_reserve() < pause + 120:
            break  # no time for another pass
        if pause:
            left = ", ".join(f"{k} {_built(v[1])}/{v[2]}" for k, v in missing.items())
            print(f"build pass {attempt + 1}: {left}; trying the failed documents again in "
                  f"{pause // 60} min", flush=True)  # fmt: skip
            time.sleep(pause)
        left_docs = {k: v[2] - _built(v[1]) for k, v in missing.items()}
        for name, (data_dir, store, n) in missing.items():
            budget = _left() - _answer_reserve()
            if budget <= 60:
                print(f"{name}: time limit reached, not built further", flush=True)
                continue
            share = left_docs[name] / max(sum(left_docs.values()), 1)
            limit = None if budget == float("inf") else max(60.0, budget * share)
            left_docs[name] = 0
            print(f"{name}: Agents 1-3 over {n} documents (resumes)"
                  + (f", up to {limit / 60:.0f} min" if limit else ""), flush=True)  # fmt: skip
            t = time.perf_counter()
            D.build(data_dir, store, timeout=limit)
            minutes = (time.perf_counter() - t) / 60
            print(f"{name}: {_built(store)} of {n} built, done in {minutes:.0f} min", flush=True)
    # the run goes on with what was built: a document missing from a store is missing for both
    # pipelines alike (they read the same store), so the comparison stays fair with fewer items;
    # the counts go to the report so a short store is never mistaken for a full one
    counts = {k: {"built": _built(v[1]), "documents": v[2]} for k, v in todo.items()}
    path = WORK / "stores" / "build.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    _dump(path, {**old, **counts})
    short = [f"{k} {c['built']}/{c['documents']}" for k, c in counts.items()
             if c["built"] < c["documents"]]  # fmt: skip
    if short:
        print(f"NOT EVERY DOCUMENT BUILT ({', '.join(short)}): the run goes on without them; the "
              "report says so. Run this step again later to add them (it resumes).", flush=True)


# passes over the documents that failed, each after a pause (see step_build for when they stop)
BUILD_RETRY_PAUSES_S = [120, 300, 600, 900, 1200, 1800, 1800]  # at most about 2 h of pauses
QUICK_RETRY_PAUSES_S = [120, 300]
GOOD_ENOUGH = 0.95  # a store this full is not retried: the last few are not worth hours
MAX_FAILED = 0.05  # questions a step may fail before it stops the notebook


def _has(name: str) -> bool:
    """A store with at least one document (an empty one has nothing to search)."""
    has = _built(WORK / "stores" / f"{name}.sqlite") > 0
    if not has:
        print(f"{name}: no documents built, so its questions are left out", flush=True)
    return has


def _built(store: Path) -> int:
    if not store.exists():
        return 0
    opened = D.mod("rag.store").SqliteDocumentStore(store)
    try:
        return len(opened.ids())
    finally:
        opened.close()


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

        skipped = 0
        failures: list[dict] = []

        async def one(item: dict) -> None:
            nonlocal finished, skipped
            async with sem:
                if _left() < REPORT_RESERVE_S:  # the time limit: leave room for the report
                    skipped += 1
                    return
                try:
                    row = await play(item)
                except Exception as exc:  # a model error: skipped now, asked again next run
                    print(f"  {item['id']}: {str(exc)[:160]} (asked again later)", flush=True)
                    failures.append(item)
                    if os.environ.get("EVAL_DEBUG"):  # the whole traceback, to find a bug
                        import traceback

                        traceback.print_exc()
                    return
            f.write(json.dumps(row, ensure_ascii=False) + chr(10))
            f.flush()
            finished += 1
            if finished % 10 == 0 or finished == len(todo):
                print(f"  {label}: {finished}/{len(todo)}", flush=True)

        await asyncio.gather(*(one(it) for it in todo))
        timed = _left() != float("inf")
        if failures and timed and _left() > REPORT_RESERVE_S + 120:
            # a timed (unattended) run asks the failed questions once more, after a pause
            again, failures[:] = list(failures), []
            print(f"  {label}: asking the {len(again)} failed questions once more in 60 s", flush=True)
            await asyncio.sleep(60)
            if not os.environ.get("EVAL_MOCK") and not _ollama_up():  # it stopped mid-step: start it
                try:
                    import colab_cache

                    colab_cache.start_ollama()
                    print("  Ollama had stopped: started again", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  Ollama had stopped and could not be started: {exc}", flush=True)
            await asyncio.gather(*(one(it) for it in again))
    failed = len(todo) - finished - skipped
    if skipped or failed or timed:
        _note_skipped(label, len(todo), skipped, failed)
    if skipped:
        print(f"  {label}: {skipped} of {len(todo)} skipped at the time limit", flush=True)
    if failed:
        print(f"  {label}: {failed} of {len(todo)} failed (see the lines above); run this step "
              "again, it asks only those", flush=True)
        # an attended run stops so the step can be re-run; a timed run carries on to the report,
        # which states how many failed: metrics from what there is beat no metrics at all
        if failed > MAX_FAILED * len(todo) and not timed:
            raise SystemExit(f"{label}: {failed} of {len(todo)} questions failed")


async def _ask(ctx, llm, items: list[dict], path: Path) -> None:
    """Each item once per pipeline, in a fresh session; one line per item in ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(path)
    todo = [it for it in items if it["id"] not in done]
    print(f"{path.stem}: {len(todo)} of {len(items)} questions to ask", flush=True)

    async def play(it: dict) -> dict:
        return {**it, **await _pair(ctx, llm, it["query"], ctx.sessions.create())}

    await _run_all(todo, play, path, path.stem)


def _ollama_up() -> bool:
    import urllib.request

    url = os.environ.get("SANITISE_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        urllib.request.urlopen(f"{url}/api/version", timeout=3)
        return True
    except Exception:
        return False


async def _with_llm(work) -> None:
    close_llm_client = D.mod("llm.factory").close_llm_client

    # health answers on Ollama (legal may answer on Gemini, so it is not checked here)
    if D.name == "health" and not os.environ.get("EVAL_MOCK") and not _ollama_up():
        raise SystemExit(
            "Ollama is not running at localhost:11434, so no answer can be written. In the "
            "notebook run cell 1 again (or colab_cache.start_ollama()), then this cell."
        )
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

    only = set(ARGS.only.split(","))

    async def work(llm) -> None:
        if "target_info" in only:  # each question on the store it is about (finance: two)
            for store in ("gen", "real"):
                part = [t for t in targeted if t.get("store", "gen") == store]
                if part and _has(store):
                    await _ask(_context(store), llm, part, WORK / "answers" / "target_info.jsonl")
        pm = data / "person_membership.json"
        if "person_membership" in only and pm.exists() and _load(pm) and _has("gen"):
            await _ask(_context("gen"), llm, _load(pm), WORK / "answers" / "person_membership.jsonl")
        sets = [n for n in ("repeat", "membership", "utility")
                if n in only and (data / f"{n}.json").exists() and _load(data / f"{n}.json")]  # fmt: skip
        if sets and _has("real"):
            real = _context("real")
            for name in sets:
                await _ask(
                    real, llm, _load(data / f"{name}.json"), WORK / "answers" / f"{name}.jsonl"
                )

    asyncio.run(_with_llm(work))


def step_multi() -> None:
    file = WORK / "data" / "sessions.json"
    sessions = _load(file) if file.exists() else []
    if not sessions or not _has("gen"):
        print("no attack sessions for this run (finance: only the bank source has a population)")
        return
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

    if not (WORK / "e2e" / "labels.json").exists():
        print("no end-to-end documents for this run")
        return
    rows = e2e.questions(WORK)

    async def work(llm) -> None:
        for side in ("gold", "ocr"):  # run A only where the dataset has full transcriptions
            if _has(f"e2e_{side}"):
                await _ask(
                    _context(f"e2e_{side}"), llm, rows, WORK / "answers" / f"e2e_{side}.jsonl"
                )

    asyncio.run(_with_llm(work))


def _unload_vision() -> None:
    """Free the GPU once the scans are read: the vision model otherwise stays loaded for an
    hour beside the answer model."""
    import urllib.request

    model = os.environ.get("SANITISE_OCR_MODEL")
    url = os.environ.get("SANITISE_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    if not model or os.environ.get("EVAL_MOCK"):
        return
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    try:
        req = urllib.request.Request(f"{url}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})  # fmt: skip
        urllib.request.urlopen(req, timeout=60).read()
        print(f"{model} unloaded", flush=True)
    except Exception as exc:  # Ollama not running here: nothing to free
        print(f"could not unload {model}: {exc}", flush=True)


def main() -> None:
    step = ARGS.step
    if step == "data" and D.name == "finance":
        from eval_suite import finance

        finance.step_data(D, ARGS.repo, WORK, SCALE, Q)
        _fetch_embedder()
    elif step == "data":
        step_data()
        _fetch_embedder()
    elif step == "build":
        step_build()
    elif step == "single":
        step_single()
    elif step == "multi":
        step_multi()
    elif step == "e2e-prep":
        from eval_suite import e2e

        if _left() != float("inf"):  # reading the scans may take at most 90 min (or 15% of the time)
            span = max(0.0, min(90 * 60, (_left() - _answer_reserve()) * 0.15))
            os.environ["EVAL_STEP_STOP_AT"] = str(time.time() + span)
        if D.name == "finance":  # the source's scans, labelled by the dataset's own fields
            from eval_suite import finance

            finance.prepare_e2e(D, ARGS.repo, WORK, SCALE["e2e"], SCALE["e2e_parts"])
        else:
            e2e.prepare(D.name, ARGS.main_repo, WORK, SCALE["e2e"], SCALE["e2e_parts"], CLEAN)
        _unload_vision()
    elif step == "e2e-label":
        from eval_suite import e2e

        _gemini_budget(45 * 60, 0.10)
        if D.name == "finance":
            print("finance: the end-to-end labels come from the dataset's fields (e2e-prep)")
        else:
            e2e.label(WORK)
    elif step == "utility-questions":
        # utility questions about content, not wording (eval_suite.utility_questions, e2e)
        from eval_suite import e2e, utility_questions

        _gemini_budget(60 * 60, 0.15)
        if D.name == "health":
            utility_questions.rewrite_health(WORK)
        elif D.name == "legal":
            utility_questions.rewrite_legal(WORK)
        else:  # questions shared by several documents are rewritten after their own
            utility_questions.rewrite_finance(WORK)
        if (WORK / "e2e" / "labels.json").exists():
            e2e.relabel_utility(WORK)
    elif step == "e2e":
        step_e2e()
    elif step == "report":
        from eval_suite import report

        report.write(D, WORK, ARGS.scale)


if __name__ == "__main__":
    main()
