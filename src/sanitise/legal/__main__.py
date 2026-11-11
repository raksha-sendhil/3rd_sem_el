"""``python -m sanitise.legal <command>``: the legal build spec from the command line.

    generate   clients, corpus, gold, non-members, training set, QA, attacks (fixed seed)
    ontology   data/ontology/{places,occupations,legal_issues}.json
    build      Agents 1-3 over the corpus into the store (resumes)
    evaluate   retrieval, both pipelines, attacks, probes and agent scores -> results.json
    report     REPORT.md and charts from results.json
    export     the dashboard bundle (play it back with SANITISE_REPLAY_DIR)
    all        every step above in order

    preflight  checks the set-up: spaCy model, domains, Gemini keys, Ollama, GPU, HF token
    check      one document through Agents 1, 2 and 3 (and one answer), step by step

The models are the legal profile's in ``config/domains.json`` (``llm_backend`` and
``gemini_model`` for Agents 1-3, ``answer_backend`` for the answers). ``--mock`` runs
everything without a model: rule pass, deterministic rewrite and canned answers.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from sanitise.config import get_settings
from sanitise.domain import settings_for
from sanitise.legal import generator, report
from sanitise.legal.export import export_bundle
from sanitise.legal.run import RunConfig, build_corpus_store, evaluate


def legal_settings():
    """Global settings with the legal profile of config/domains.json applied (its models)."""
    return settings_for(get_settings(), "legal")


COMMANDS = ("generate", "ontology", "build", "evaluate", "report", "export", "all")
EXTRA = ("preflight", "check")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sanitise.legal", description=__doc__.split("\n")[0])
    p.add_argument("command", choices=COMMANDS + EXTRA)
    p.add_argument("--data", default="data/legal")
    p.add_argument("--store", default=None, help="default: <data>/store.sqlite")
    p.add_argument("--out", default=None, help="results folder (default: <data>/results)")
    p.add_argument("--bundle", default=None, help="export folder (default: <data>/bundle)")
    p.add_argument("--ontology", default="data/ontology")
    p.add_argument("--seed", type=int, default=generator.DEFAULT_SEED)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--retrieval", choices=("hybrid", "bm25"), default=None)
    p.add_argument("--limit", type=int, default=None, help="only the first N corpus documents")
    p.add_argument("--mi-limit", type=int, default=100)
    p.add_argument("--mock", action="store_true", help="no model: rules + canned answers")
    p.add_argument("--doc", default="c001_intake_note", help="check: corpus file name")
    p.add_argument("--question", default=None, help="check: also answer this")
    return p


async def _build(cfg: RunConfig, mock: bool) -> None:
    from sanitise.llm.factory import build_agent1_llm, build_llm_client, close_llm_client

    def progress(i: int, doc: object, status: str) -> None:
        if status != "skipped":
            print(f"  [{i}] {getattr(doc, 'source', '')}: {status}", flush=True)

    if mock:
        from sanitise.mock_mode import mock_llm, mock_sanitise

        report_ = await build_corpus_store(
            cfg, mock_llm(), sanitiser=mock_sanitise, progress=progress
        )
    else:
        settings = legal_settings()
        llm = build_llm_client(settings)
        detector = build_agent1_llm(settings, llm)
        try:
            report_ = await build_corpus_store(cfg, llm, detector_llm=detector, progress=progress)
        finally:
            await close_llm_client(llm)
    print(report_.summary())


async def _evaluate(cfg: RunConfig, mock: bool) -> None:
    from sanitise.llm.factory import build_answer_llm, build_llm_client, close_llm_client

    if mock:
        from sanitise.mock_mode import mock_llm

        results = await evaluate(cfg, mock_llm())
    else:
        settings = legal_settings()
        agents = build_llm_client(settings)
        answer_llm = build_answer_llm(settings, agents)
        try:
            results = await evaluate(cfg, answer_llm)
        finally:
            await close_llm_client(answer_llm)
            if answer_llm is not agents:
                await close_llm_client(agents)
    q = results["answer_quality"]
    print(
        f"QA accuracy baseline {q['baseline_accuracy']:.2f} protected "
        f"{q['protected_accuracy']:.2f}; results in {cfg.out_dir}"
    )


def _status(ok: bool | None, label: str, detail: str = "") -> bool:
    mark = {True: "OK  ", False: "FAIL", None: "--  "}[ok]
    print(f"  [{mark}] {label}" + (f": {detail}" if detail else ""), flush=True)
    return ok is not False


async def _ollama_health(settings) -> dict:
    from sanitise.llm.ollama import OllamaClient

    async with OllamaClient.from_settings(settings) as ollama:
        return await ollama.health()


def _preflight(data: Path) -> bool:
    """What a run needs, checked one by one. Returns False if a required check failed."""
    import importlib.util
    import os
    import shutil
    import subprocess

    from sanitise.domain import profiles
    from sanitise.llm.gemini import fingerprint, gemini_api_keys

    settings = legal_settings()
    ok = True
    print("Code and data")
    has_spacy = importlib.util.find_spec("en_core_web_lg") is not None
    ok &= _status(
        has_spacy,
        "spaCy en_core_web_lg",
        "" if has_spacy else "python -m spacy download en_core_web_lg",
    )
    n = len(list((data / "corpus").glob("*.json")))
    ok &= _status(n == 300, "legal corpus", f"{n} documents in {data / 'corpus'}")
    ok &= _status(Path("data/ontology/places.json").is_file(), "ontology tables", "data/ontology/")

    print("Domains (config/domains.json)")
    for name in profiles():
        ds = settings_for(get_settings(), name)
        answer = ds.gemini_model if ds.answer_backend == "gemini" else ds.answer_backend
        _status(True, name, f"agents {ds.model}, answers {answer}, agent 1 {ds.agent1_backend}")

    agents_on_gemini = settings.llm_backend == "gemini"
    print("Gemini (agents and answers)" if agents_on_gemini else "Gemini (answers)")
    keys = gemini_api_keys(settings.gemini_api_key, settings.gemini_api_keys)
    needs_gemini = agents_on_gemini or settings.answer_backend == "gemini"
    fingerprints = ", ".join(fingerprint(k) for k in keys)
    _status(
        True if keys else (False if needs_gemini else None),
        f"{len(keys)} key(s)",
        fingerprints or "add GEMINI_API_KEY (Kaggle secret or .env)",
    )
    ok &= bool(keys) or not needs_gemini
    has_genai = importlib.util.find_spec("google") is not None and (
        importlib.util.find_spec("google.genai") is not None
    )
    _status(has_genai or None, "google-genai", "" if has_genai else "pip install google-genai")

    if agents_on_gemini:
        _status(None, "raw text", "Agents 1-3 send originals to Gemini: fictional data only")
    else:
        print(f"Ollama (Agents 2-3: {settings.model})")
        health = asyncio.run(_ollama_health(settings))
        ok &= _status(health["reachable"], "server", settings.ollama_base_url)
        ok &= _status(health["model_available"], f"model {settings.model}", health["detail"] or "")

    print("GPU and Agent 1")
    gpus = "none"
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        gpus = "; ".join(out.stdout.strip().splitlines()) or "none"
    _status(True if gpus != "none" else None, "GPUs", gpus)
    if settings.agent1_backend == "transformers":
        adapter = settings.agent1_adapter
        ok &= _status(bool(adapter) and Path(adapter).is_dir(), "LoRA adapter", str(adapter))
    else:
        through = "Gemini" if agents_on_gemini else "Ollama"
        _status(None, "Agent 1 LLM pass", f"base model through {through} ({settings.model})")
    _status(
        True if os.environ.get("HF_TOKEN") else None,
        "HF_TOKEN",
        "needed only for fine-tuning (google/gemma-4-E4B-it is gated)",
    )
    print("All required checks passed." if ok else "Fix the FAIL lines above.")
    return bool(ok)


async def _check(data: Path, name: str, mock: bool, question: str | None) -> None:
    """One corpus document through Agents 1, 2 and 3, printed so a person can check each."""
    import json

    from sanitise.domain import use_domain
    from sanitise.llm.factory import (
        build_agent1_llm,
        build_answer_llm,
        build_llm_client,
        close_llm_client,
    )
    from sanitise.rag.answer_prompt import build_answer_prompt

    doc = json.loads((data / "corpus" / f"{name}.json").read_text(encoding="utf-8"))
    gold = json.loads((data / "gold" / f"{name}.json").read_text(encoding="utf-8"))
    settings = legal_settings()
    bar = "-" * 78
    with use_domain("legal"):
        if mock:
            from sanitise.mock_mode import mock_llm, mock_sanitise

            agents = answerer = mock_llm()
            record = await mock_sanitise(doc["doc_id"], doc["text"], agents)
        else:
            from sanitise.agents.pipeline import sanitise

            agents = build_llm_client(settings)
            answerer = build_answer_llm(settings, agents)
            record = await sanitise(
                doc["doc_id"],
                doc["text"],
                agents,
                detector_llm=build_agent1_llm(settings, agents),
            )
        found = [(s.start, s.end) for s in record.detector.spans if s.start is not None]
        missed = [
            g["text"]
            for g in gold["spans"]
            if g["start"] is not None and not any(a < g["end"] and g["start"] < b for a, b in found)
        ]
        print(f"{bar}\nORIGINAL ({name})\n{bar}\n{doc['text']}\n")
        print(f"{bar}\nAGENT 1: {len(record.detector.spans)} spans (rules + LLM pass + merge)")
        print(bar)
        for sp in record.detector.spans:
            where = f"{sp.start}-{sp.end}" if sp.start is not None else "combination"
            print(f"  {sp.category.value:16} {sp.text!r:48} {sp.method:4} {where}")
        print(f"  gold spans missed: {missed or 'none'}\n")
        print(f"{bar}\nAGENT 2: {len(record.extractor.facts)} facts\n{bar}")
        for f in record.extractor.facts:
            print(f"  {f.subject:15} {f.relation:20} {f.importance.value:6} {f.value!r}")
        print(f"\n{bar}\nAGENT 3: routing\n{bar}")
        for d in record.rewriter.decisions:
            print(
                f"  {d.category.value:16} {d.action.value:11} {d.span_text!r:40} "
                f"-> {d.replacement!r}"
            )
        print(f"\n{bar}\nAGENT 3: safe text (all the answer model ever sees)\n{bar}")
        print(record.rewriter.safe_text)
        print(f"\nleak check: {record.residual_leaks or 'no residual leaks'}")
        print(f"fallbacks used: {record.meta.fallback or 'none'}")
        meta = record.meta
        print(
            f"time: agent 1 {meta.detect_ms} ms, agent 2 {meta.extract_ms} ms, "
            f"agent 3 {meta.rewrite_ms} ms"
        )
        if question:
            system, user = build_answer_prompt(
                question, [(doc["doc_id"], record.rewriter.safe_text)]
            )
            text = await answerer.text_call(system, user)
            model = getattr(answerer, "model", "")
            print(f"\n{bar}\nANSWER from {model} to: {question}\n{bar}\n{text}")
        if not mock:
            await close_llm_client(agents)
            if answerer is not agents:
                await close_llm_client(answerer)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    data = Path(args.data)
    cfg = RunConfig(
        data_dir=data,
        store_path=Path(args.store) if args.store else data / "store.sqlite",
        out_dir=Path(args.out) if args.out else data / "results",
        k=args.k,
        retrieval_mode=args.retrieval or ("bm25" if args.mock else get_settings().retrieval_mode),
        index_dir=data / "index",
        embedding_model=get_settings().embedding_model,
        mi_limit=args.mi_limit,
        limit_docs=args.limit,
    )
    if args.command == "preflight":
        raise SystemExit(0 if _preflight(data) else 1)
    if args.command == "check":
        asyncio.run(_check(data, args.doc, args.mock, args.question))
        return
    steps = COMMANDS[:-1] if args.command == "all" else (args.command,)
    for step in steps:
        print(f"== {step}", flush=True)
        if step == "generate":
            generator.generate(data, args.seed)
        elif step == "ontology":
            generator.write_ontology(args.ontology)
        elif step == "build":
            asyncio.run(_build(cfg, args.mock))
        elif step == "evaluate":
            asyncio.run(_evaluate(cfg, args.mock))
        elif step == "report":
            print(report.write_report(cfg.out_dir / "results.json", cfg.out_dir))
        elif step == "export":
            print(export_bundle(data, cfg.store_path, cfg.out_dir, args.bundle or data / "bundle"))


if __name__ == "__main__":
    main()
