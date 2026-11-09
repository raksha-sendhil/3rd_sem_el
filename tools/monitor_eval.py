"""Metric 2, Secret Recovery Rate, and the session monitor's over-blocking check.

    python tools/monitor_eval.py                       # answers on the configured model (Ollama)
    python tools/monitor_eval.py --answers gemini      # answers on the Gemini client of Agents 2-3
    python tools/monitor_eval.py --mock                # no model, no store: fixtures + sample attacks

**Attacks.** Plays every attack script in ``data/attacks`` (stepwise, context dump, interception)
against both pipelines, one question per round in one session, with the session monitor on
(docs/session_monitor.md). Every round is scored three ways (attacks/runner.py):

* Standard RAG (the originals),
* rewriting alone (the protected draft before the monitor),
* rewriting + session monitor (what the protected pipeline actually answered).

Secret Recovery Rate (Sun et al., 2026): the share of the target's private fields the attacker
has recovered after each round. Also the candidate pool (people in the person table still
consistent with what was disclosed) and the round at which the target is identified.

**Over-blocking.** The harmless questions in ``data/monitor/benign_questions.json``, asked on
the protected pipeline twice: each in a fresh session, and all in one session (a clinician
asking many questions in a row). Every answer the monitor generalises or blocks counts against
it: a monitor that refuses everything would score perfectly on the attacks.

Needs a built store (``tools/build_store.py``), like Target Info. Writes
``results/monitor/<run>/``: ``report.md``, ``report.json`` and ``rounds.jsonl`` (every round,
every answer and verdict).
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

from sanitise.attacks.runner import load_attacks, load_people, run_attack  # noqa: E402
from sanitise.config import get_settings  # noqa: E402
from sanitise.llm.factory import agents_llm, build_llm_client, close_llm_client  # noqa: E402
from sanitise.mock_mode import SAMPLE_DIR, build_mock_context, mock_llm  # noqa: E402
from sanitise.monitor import build_monitor  # noqa: E402
from sanitise.rag.answer import DEFAULT_K, RagContext, answer  # noqa: E402
from sanitise.rag.dense import build_retriever  # noqa: E402
from sanitise.rag.store import SqliteDocumentStore  # noqa: E402
from sanitise.schemas import AttackRun  # noqa: E402

SETUPS = {
    "baseline": "Standard RAG",
    "unmonitored": "Rewriting alone",
    "protected": "Rewriting + session monitor",
}


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
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


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def attack_rows(run: AttackRun, kind: str) -> list[dict]:
    rows = []
    for r in run.rounds:
        row = {"attack_id": run.attack_id, "kind": kind, "round": r.round, "query": r.query,
               "verdict": r.protected.monitor.verdict, "reason": r.protected.monitor.reason,
               "baseline_answer": r.baseline.answer, "protected_answer": r.protected.answer,
               "draft_answer": r.protected.draft_answer}  # fmt: skip
        for setup in SETUPS:
            leak = getattr(r, f"{setup}_leakage")
            row[f"{setup}_srr"] = leak.recovery_rate if leak else None
            row[f"{setup}_pool"] = leak.candidate_pool if leak else None
            row[f"{setup}_identified"] = bool(leak and leak.identified)
        rows.append(row)
    return rows


def summarise(runs: list[tuple[AttackRun, str]], benign: dict) -> dict:
    per_attack = []
    for run, kind in runs:
        last = run.rounds[-1]
        entry = {"attack_id": run.attack_id, "kind": kind, "rounds": len(run.rounds),
                 "verdicts": [r.protected.monitor.verdict for r in run.rounds]}  # fmt: skip
        for setup in SETUPS:
            leak = getattr(last, f"{setup}_leakage")
            entry[f"{setup}_final_srr"] = leak.recovery_rate if leak else None
            entry[f"{setup}_final_pool"] = leak.candidate_pool if leak else None
            entry[f"{setup}_identified_at"] = getattr(run, f"{setup}_identified_at")
        per_attack.append(entry)
    overall: dict = {}
    for kind in sorted({k for _, k in runs}) + ["all"]:
        chosen = [e for e in per_attack if kind == "all" or e["kind"] == kind]
        overall[kind] = {"attacks": len(chosen)}
        for setup in SETUPS:
            srrs = [e[f"{setup}_final_srr"] for e in chosen if e[f"{setup}_final_srr"] is not None]
            overall[kind][setup] = {
                "mean_final_srr": sum(srrs) / len(srrs) if srrs else None,
                "identified": sum(e[f"{setup}_identified_at"] is not None for e in chosen),
            }
    return {"per_attack": per_attack, "overall": overall, "benign": benign}


def report_md(meta: dict, summary: dict) -> str:
    out = [f"# Session monitor evaluation - {meta['run']}", "",
           f"Answer model {meta['answer_model']}, store `{meta['store']}` ({meta['documents']} documents), "
           f"monitor k = {meta['k']}, at most {meta['max_attributes']} kinds of detail per person, "
           f"commit {meta['commit']}, {meta['finished']}.", "",
           "**Secret Recovery Rate** (Sun et al., 2026): the share of the target's private fields the "
           "attacker has recovered by the last round; lower is better. **Identified**: attacks in which "
           "the target was singled out (name or phone disclosed, or one person left in the pool).", "",
           "## Attacks", "", "| Kind | Attacks | " + " | ".join(
               f"{label}: SRR / identified" for label in SETUPS.values()) + " |",
           "|---|---|" + "---|" * len(SETUPS)]  # fmt: skip
    for kind, o in summary["overall"].items():
        cells = [f"{_pct(o[s]['mean_final_srr'])} / {o[s]['identified']} of {o['attacks']}" for s in SETUPS]
        name = "**all**" if kind == "all" else kind
        out.append(f"| {name} | {o['attacks']} | " + " | ".join(cells) + " |")
    out += ["", "## Per attack", "",
            "| Attack | Rounds | Monitor verdicts | " + " | ".join(
                f"{label} (SRR, identified at)" for label in SETUPS.values()) + " |",
            "|---|---|---|" + "---|" * len(SETUPS)]  # fmt: skip
    for e in summary["per_attack"]:
        cells = [
            f"{_pct(e[f'{s}_final_srr'])}, {('Q' + str(e[f'{s}_identified_at'])) if e[f'{s}_identified_at'] else 'never'}"
            for s in SETUPS
        ]
        out.append(f"| {e['attack_id']} | {e['rounds']} | {' '.join(e['verdicts'])} | " + " | ".join(cells) + " |")
    b = summary["benign"]
    if b:
        out += ["", "## Over-blocking: harmless questions on the protected pipeline", "",
                "**Refused** is the over-blocking cost: a harmless question gets no answer. **Generalised** "
                "still answers, with its personal details made less specific (an answer that recites a "
                "record's age or town about one person is generalised whatever the question was).", "",
                "| Asked | Questions | Allowed | Generalised | Refused (blocked) |",
                "|---|---|---|---|---|"]  # fmt: skip
        for mode, label in (("fresh", "each in a fresh session"), ("one_session", "all in one session")):
            c = b[mode]
            n = sum(c.values())
            def share(v: int, n: int = n) -> str:
                return _pct(v / n if n else None)

            out.append(f"| {label} | {n} | {share(c.get('allowed', 0))} | {share(c.get('generalised', 0))} | "
                       f"**{share(c.get('blocked', 0))}** |")  # fmt: skip
        if b.get("stepped_in"):
            out += ["", "Questions the monitor stepped in on:", ""]
            out += [f"* ({x['mode']}) {x['question']}: {x['verdict']}, {x['reason']}" for x in b["stepped_in"]]
    return "\n".join(out) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", choices=["ollama", "gemini", "mock"], default="ollama")
    ap.add_argument("--mock", action="store_true", help="fixtures, sample attacks and a mock model")
    ap.add_argument("--store", default=None, help="default: settings.store_path")
    ap.add_argument("--attacks", default=None, help="default: settings.attacks_dir")
    ap.add_argument("--people", default=None, help="default: settings.people_path")
    ap.add_argument("--kinds", default="stepwise,context_dump,interception")
    ap.add_argument("--benign", default=str(ROOT / "data" / "monitor" / "benign_questions.json"))
    ap.add_argument("--no-benign", action="store_true")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="documents retrieved per question")
    ap.add_argument("--out", default=str(ROOT / "results" / "monitor"))
    ap.add_argument("--run", default=None)
    args = ap.parse_args()

    settings = get_settings().model_copy(update={"monitor": "ledger", "monitor_db_path": ":memory:"})
    if args.mock:
        llm, answer_model = mock_llm(), "mock (quotes the first retrieved record)"
        ctx = await build_mock_context(llm)
        attacks_dir, people_path = SAMPLE_DIR / "attacks", SAMPLE_DIR / "people.json"
        store_name = "data/fixtures (mock)"
    else:
        store = SqliteDocumentStore(args.store or settings.store_path)
        if not store.ids():
            raise SystemExit(f"the store {store.path} is empty: run tools/build_store.py first")
        retriever = build_retriever(store, mode=settings.retrieval_mode, index_dir=settings.index_dir,
                                    model_name=settings.embedding_model)  # fmt: skip
        ctx = RagContext(store=store, retriever=retriever)
        llm, answer_model = _answer_client(args.answers)
        attacks_dir = Path(args.attacks or settings.attacks_dir)
        people_path = Path(args.people or settings.people_path)
        store_name = str(store.path)
    ctx.monitor = build_monitor(settings, ctx.store)
    people = load_people(people_path)
    kinds = set(args.kinds.split(","))
    attacks = [a for a in load_attacks(attacks_dir).values() if a.kind in kinds]
    in_store = set(ctx.store.ids())
    by_id = {p.person_id: p for p in people}
    attacks = [a for a in attacks if a.target_person_id in by_id
               and set(by_id[a.target_person_id].doc_ids or []) <= in_store]  # fmt: skip
    run_name = args.run or (f"mock_{len(in_store)}docs" if args.mock else f"{args.answers}_{len(in_store)}docs")
    out = Path(args.out) / run_name
    out.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[AttackRun, str]] = []
    rows: list[dict] = []
    benign: dict = {}
    try:
        for i, attack in enumerate(attacks, 1):
            run = await run_attack(ctx, attack, people, llm)
            runs.append((run, attack.kind))
            rows += attack_rows(run, attack.kind)
            verdicts = " ".join(r.protected.monitor.verdict for r in run.rounds)
            print(f"[{i}/{len(attacks)}] {attack.attack_id}: {verdicts}", flush=True)
        if not args.no_benign:
            questions = json.loads(Path(args.benign).read_text(encoding="utf-8"))["questions"]
            counts = {"fresh": Counter(), "one_session": Counter()}
            stepped = []
            one = ctx.sessions.create()
            for q in questions:
                for mode, session in (("fresh", ctx.sessions.create()), ("one_session", one)):
                    a = await answer(ctx, q, "protected", session, llm, args.k)
                    counts[mode][a.monitor.verdict] += 1
                    if a.monitor.verdict != "allowed":
                        stepped.append({"mode": mode, "question": q, "verdict": a.monitor.verdict,
                                        "reason": a.monitor.reason})  # fmt: skip
            print(f"harmless questions: fresh {dict(counts['fresh'])}, one session {dict(counts['one_session'])}")
            benign = {"fresh": dict(counts["fresh"]), "one_session": dict(counts["one_session"]),
                      "stepped_in": stepped}  # fmt: skip
    finally:
        await close_llm_client(llm)

    summary = summarise(runs, benign)
    meta = {"run": run_name, "finished": datetime.now().isoformat(timespec="seconds"), "commit": _commit(),
            "answer_model": answer_model, "store": store_name, "documents": len(in_store),
            "k": settings.monitor_k, "max_attributes": settings.monitor_max_attributes,
            "max_climb": settings.monitor_max_climb, "retrieval_k": args.k}  # fmt: skip
    (out / "rounds.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (out / "report.json").write_text(json.dumps({"meta": meta, **summary}, indent=1), encoding="utf-8")
    md = report_md(meta, summary)
    (out / "report.md").write_text(md, encoding="utf-8")
    print("\n" + md + f"\nsaved {out / 'report.md'}")


if __name__ == "__main__":
    asyncio.run(main())
