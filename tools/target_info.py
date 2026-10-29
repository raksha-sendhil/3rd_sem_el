"""Metric 1, Target Info: play Person D's fixed adversarial query set against both pipelines.

    python tools/target_info.py                      # answers on the configured model (Ollama)
    python tools/target_info.py --answers gemini     # answers on the Gemini client of Agents 2-3
    python tools/target_info.py --answers mock       # no model at all (checks the plumbing)

Needs a built store (``sanitise build --input data/corpus/docs``): the baseline answers from
the original documents, the protected pipeline from Agents 1-3's rewrites of the same
documents, with the same retrieval (top-3 over the originals) and the same answer model and
settings. Each query is asked once, in a fresh session, so the metric measures what a single
query leaks (the offline rewriting stage), not what a session accumulates.

Target Info (``evaluation.attacks.target_info``): the number of distinct true private values
(a name, phone, age, town, occupation or condition of any of the 120 people) that appear in an
answer and were not already in that answer's query. Lower is better; zero is ideal.

Writes ``results/target_info/<run>/``: ``answers.jsonl`` (every query, both answers, the
documents retrieved), ``report.json`` (the numbers, every leaked value and the run's settings)
and ``report.md`` (the same for people). A stopped run resumes from ``answers.jsonl``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sanitise.attacks.runner import load_people  # noqa: E402
from sanitise.config import get_settings  # noqa: E402
from sanitise.evaluation.attacks import leaked_values  # noqa: E402
from sanitise.llm.factory import agents_llm, build_llm_client, close_llm_client  # noqa: E402
from sanitise.mock_mode import mock_llm  # noqa: E402
from sanitise.rag.answer import DEFAULT_K, RagContext, answer  # noqa: E402
from sanitise.rag.dense import build_retriever  # noqa: E402
from sanitise.rag.store import SqliteDocumentStore  # noqa: E402
from sanitise.schemas import RagAnswer  # noqa: E402

PIPELINES = ("baseline", "protected")


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _answer_client(kind: str):
    settings = get_settings()
    if kind == "mock":
        return mock_llm(), "mock (quotes the first retrieved record)"
    llm = build_llm_client(settings, backend="ollama")
    if kind == "gemini":
        if settings.agents_backend != "gemini":
            raise SystemExit("--answers gemini needs SANITISE_AGENTS_BACKEND=gemini")
        return agents_llm(llm), f"{settings.agents_model} (Gemini API)"
    return llm, f"{settings.model} (Ollama)"


def _score(people, rows: list[dict], pipeline: str, kind: str | None = None) -> list[dict]:
    answers = [
        RagAnswer.model_validate(r[pipeline]) for r in rows if kind is None or r["kind"] == kind
    ]
    return leaked_values(people, answers)


def report(rows: list[dict], people, meta: dict, out: Path) -> dict:
    kinds = sorted({r["kind"] for r in rows})
    result: dict = {"meta": meta, "queries": len(rows), "pipelines": {}}
    for pipeline in PIPELINES:
        leaked = _score(people, rows, pipeline)
        result["pipelines"][pipeline] = {
            "target_info": len(leaked),
            "by_field": dict(Counter(v["field"] for v in leaked)),
            "by_query_kind": {k: len(_score(people, rows, pipeline, k)) for k in kinds},
            "leaked": leaked,
        }
    (out / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(_markdown(result, kinds), encoding="utf-8")
    return result


def _markdown(result: dict, kinds: list[str]) -> str:
    meta, p = result["meta"], result["pipelines"]
    base, prot = p["baseline"], p["protected"]
    fields = sorted(set(base["by_field"]) | set(prot["by_field"]))
    lines = [
        "# Target Info (Metric 1)",
        "",
        f"Run `{meta['run']}`, {meta['finished']}, commit `{meta['commit']}`.",
        "",
        f"{result['queries']} adversarial queries ({meta['query_kinds']}) from "
        f"`{meta['queries_file']}`, each asked once in a fresh session against both pipelines; "
        f"{meta['people']} people, {meta['store_documents']} documents in the store.",
        "",
        "Target Info = distinct true private values (name, phone, age, town, occupation, "
        "condition of any person) in an answer that its query did not already contain. "
        "Lower is better; zero is ideal.",
        "",
        "| | Unprotected RAG (baseline) | Our pipeline (protected) |",
        "|---|---:|---:|",
        f"| **Target Info** | **{base['target_info']}** | **{prot['target_info']}** |",
    ]
    lines += [
        f"| from `{k}` queries | {base['by_query_kind'][k]} | {prot['by_query_kind'][k]} |"
        for k in kinds
    ]
    lines += [
        f"| {f} | {base['by_field'].get(f, 0)} | {prot['by_field'].get(f, 0)} |" for f in fields
    ]
    lines += [
        "",
        "## Settings",
        "",
        f"* Answer model (both pipelines): {meta['answer_model']}",
        f"* Agents 1-3 that built the protected store: {meta['agents']}",
        f"* Retrieval: {meta['retrieval']}, top {meta['k']} documents",
        f"* Store: `{meta['store']}`",
    ]
    for pipeline in PIPELINES:
        leaked = p[pipeline]["leaked"]
        lines += ["", f"## Values leaked by the {pipeline} pipeline ({len(leaked)})", ""]
        lines += [
            f"* {v['field']}: **{v['value']}** ({v['person_id']}) - query: {v['query']}"
            for v in leaked
        ] or ["(none)"]
    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace, queries: list[dict]) -> None:
    settings = get_settings()
    people = load_people(args.people)
    store = SqliteDocumentStore(args.store or settings.store_path)
    if not store.ids():
        raise SystemExit(f"the store {store.path} is empty: run `sanitise build` first")
    retriever = build_retriever(
        store,
        mode=args.retrieval or settings.retrieval_mode,
        index_dir=settings.index_dir,
        model_name=settings.embedding_model,
    )
    ctx = RagContext(store=store, retriever=retriever)
    llm, answer_model = _answer_client(args.answers)

    out = Path(args.out) / args.run
    out.mkdir(parents=True, exist_ok=True)
    answers_path = out / "answers.jsonl"
    rows = []
    if answers_path.exists():
        rows = [json.loads(line) for line in answers_path.read_text(encoding="utf-8").splitlines()]
    done = {r["query_id"] for r in rows}
    try:
        with answers_path.open("a", encoding="utf-8") as f:
            for i, q in enumerate(queries, 1):
                if q["query_id"] in done:
                    continue
                session = ctx.sessions.create()
                row = {"query_id": q["query_id"], "kind": q["kind"], "person_id": q["person_id"]}
                for pipeline in PIPELINES:
                    a = await answer(ctx, q["query"], pipeline, session, llm, args.k)
                    row[pipeline] = a.model_dump(mode="json")
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                rows.append(row)
                print(f"[{i}/{len(queries)}] {q['query_id']} {q['kind']}", flush=True)
    finally:
        await close_llm_client(llm)

    order = {q["query_id"]: n for n, q in enumerate(queries)}
    rows = sorted((r for r in rows if r["query_id"] in order), key=lambda r: order[r["query_id"]])
    sample = store.sanitised(next(iter(store.ids())))
    meta = {
        "run": args.run,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "commit": _commit(),
        "queries_file": args.queries,
        "query_kinds": ", ".join(
            f"{n} {k}" for k, n in Counter(q["kind"] for q in queries).items()
        ),
        "people": len(people),
        "store": str(store.path),
        "store_documents": len(store.ids()),
        "answer_model": answer_model,
        "agents": (
            f"rules + {settings.agents_model} (Gemini API)"
            if settings.agents_backend == "gemini"
            else sample.meta.model
        ),
        "retrieval": type(retriever).__name__,
        "k": args.k,
    }
    result = report(rows, people, meta, out)
    p = result["pipelines"]
    print(
        f"Target Info over {len(rows)} queries: baseline {p['baseline']['target_info']}, "
        f"protected {p['protected']['target_info']}  ->  {out / 'report.md'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", default="data/attacks/target_info_queries.json")
    parser.add_argument("--people", default="data/corpus/people.json")
    parser.add_argument("--store", help="SQLite store (default: settings.store_path)")
    parser.add_argument("--answers", choices=["ollama", "gemini", "mock"], default="ollama")
    parser.add_argument("--retrieval", choices=["hybrid", "bm25"])
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default="results/target_info")
    parser.add_argument("--run", default=datetime.now().strftime("%Y-%m-%d_%H%M"))
    args = parser.parse_args()
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))[: args.limit]
    asyncio.run(run(args, queries))
