"""``python -m sanitise.legal <command>``: the legal build spec from the command line.

    generate   clients, corpus, gold, non-members, training set, QA, attacks (fixed seed)
    ontology   data/ontology/{places,occupations,legal_issues}.json
    build      Agents 1-3 over the corpus into the store (resumes)
    evaluate   retrieval, both pipelines, attacks, probes and agent scores -> results.json
    report     REPORT.md and charts from results.json
    export     the dashboard bundle (play it back with SANITISE_REPLAY_DIR)
    all        every step above in order

The models are the legal profile's in ``config/domains.json`` (``model`` for Agents 2-3
through Ollama, ``answer_backend``/``gemini_model`` for the answers, ``agent1_backend`` and
``agent1_adapter`` for the fine-tuned Agent 1). ``--mock`` runs
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


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sanitise.legal", description=__doc__.split("\n")[0])
    p.add_argument("command", choices=COMMANDS)
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
