"""The ``sanitise`` command-line tool.

    sanitise detect --input data/fixtures/case_07.json --verbose
    sanitise detect --input data/fixtures/ --out out/            # batch
    sanitise score  --gold data/gold/ --pred out/
    sanitise health
    sanitise serve  --port 8000

Exit codes: 0 success, 1 pipeline error (the spec 8.5 envelope is printed),
2 usage error, 3 a stage that is not implemented yet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from sanitise.agents.detector.pipeline import detect_with_trace
from sanitise.cli import render
from sanitise.config import Settings, get_settings
from sanitise.errors import PipelineError
from sanitise.evaluation import scoring
from sanitise.llm.factory import build_llm_client, close_llm_client
from sanitise.llm.ollama import OllamaClient
from sanitise.loader import Document, iter_documents, load_document
from sanitise.results import load_gold, load_predictions, save_result
from sanitise.schemas import PIPELINE_VERSION, DetectorResult

app = typer.Typer(
    name="sanitise",
    help="Agent 1 - Sensitive Information Detector (Phase 2).",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

out = Console()
err = Console(stderr=True)

EXIT_PIPELINE_ERROR = 1
EXIT_NOT_IMPLEMENTED = 3


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        handlers=[RichHandler(console=err, show_path=False, show_time=False)],
        force=True,
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _version(value: bool) -> None:
    if value:
        out.print(f"sanitise {PIPELINE_VERSION}")
        raise typer.Exit()


@app.callback()
def main(
    log_level: str = typer.Option(None, "--log-level", help="DEBUG, INFO, WARNING, ERROR."),
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True),
) -> None:
    _setup_logging(log_level or get_settings().log_level)


def _not_implemented(exc: NotImplementedError) -> typer.Exit:
    err.print(f"[bold yellow]not implemented[/] {exc}")
    return typer.Exit(EXIT_NOT_IMPLEMENTED)


# ------------------------------------------------------------------ detect


def _collect_documents(
    path: Path, index: int | None, settings: Settings, skip_too_long: bool
) -> tuple[list[Document], bool]:
    """(documents, is_batch)."""
    if index is not None:
        return [load_document(path, index=index, settings=settings)], False
    docs = list(iter_documents(path, settings=settings, skip_too_long=skip_too_long))
    return docs, path.is_dir() or len(docs) != 1


async def _run_detect(
    docs: list[Document],
    *,
    settings: Settings,
    backend: str,
    model: str | None,
    mock_responses: Path | None,
    run_llm: bool,
    concurrency: int,
) -> list[tuple[Document, tuple[DetectorResult, object] | PipelineError]]:
    llm = build_llm_client(settings, backend=backend, model=model, mock_responses=mock_responses)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(doc: Document) -> tuple[DetectorResult, object] | PipelineError:
        async with semaphore:
            try:
                return await detect_with_trace(
                    doc.doc_id, doc.text, llm, run_llm=run_llm, max_chars=settings.max_doc_chars
                )
            except PipelineError as exc:
                return exc

    try:
        results = await asyncio.gather(*(one(d) for d in docs))
    finally:
        await close_llm_client(llm)
    return list(zip(docs, results, strict=True))


@app.command()
def detect(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        help="A fixture/ChatDoctor/Enron file, or a directory of them.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Pretty-print each stage, as in the demo."
    ),
    out_dir: Path | None = typer.Option(
        None, "--out", "-o", help="Write <doc_id>.json results here (for `score`)."
    ),
    index: int | None = typer.Option(
        None, "--index", help="Record index inside a multi-record file."
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Ollama model (default from settings)."
    ),
    mock: bool = typer.Option(False, "--mock", help="Use MockLLM instead of Ollama."),
    mock_responses: Path | None = typer.Option(
        None,
        "--mock-responses",
        exists=True,
        dir_okay=False,
        help='JSON {"QuasiList": [reply, ...]} scripted MockLLM replies (implies --mock).',
    ),
    rules_only: bool = typer.Option(False, "--rules-only", help="Skip the LLM pass."),
    as_json: bool = typer.Option(
        False, "--json", help="Print DetectorResult JSON even for batches."
    ),
    concurrency: int = typer.Option(
        1, "--concurrency", "-j", min=1, help="Documents in flight at once."
    ),
    skip_too_long: bool = typer.Option(
        False, "--skip-too-long", help="In batches, skip documents over the length limit."
    ),
) -> None:
    """Run Agent 1 on one document or a batch."""
    settings = get_settings()
    try:
        docs, is_batch = _collect_documents(input_path, index, settings, skip_too_long)
    except PipelineError as exc:
        render.render_error(err, exc, str(input_path))
        raise typer.Exit(EXIT_PIPELINE_ERROR) from None
    except (ValueError, IndexError, json.JSONDecodeError) as exc:
        err.print(f"[bold red]error[/] {exc}")
        raise typer.Exit(2) from None
    if not docs:
        err.print(f"[bold red]error[/] no documents found in {input_path}")
        raise typer.Exit(2)

    backend = "mock" if (mock or mock_responses) else settings.llm_backend
    try:
        outcomes = asyncio.run(
            _run_detect(
                docs,
                settings=settings,
                backend=backend,
                model=model,
                mock_responses=mock_responses,
                run_llm=not rules_only,
                concurrency=concurrency,
            )
        )
    except NotImplementedError as exc:
        raise _not_implemented(exc) from None

    failures = 0
    for doc, outcome in outcomes:
        if isinstance(outcome, PipelineError):
            failures += 1
            render.render_error(err, outcome, doc.source)
            continue
        result, trace = outcome
        if out_dir is not None:
            save_result(result, out_dir)
        if verbose:
            render.render_verbose(out, doc, result, trace)  # type: ignore[arg-type]
        elif is_batch and not as_json:
            render.render_summary_line(out, doc, result)
        else:
            sys.stdout.write(result.model_dump_json(indent=None if is_batch else 2) + "\n")

    if is_batch or out_dir is not None:
        ok = len(outcomes) - failures
        where = f" -> {out_dir}" if out_dir is not None else ""
        err.print(f"[dim]{ok}/{len(outcomes)} documents processed{where}[/]")
    if failures:
        raise typer.Exit(EXIT_PIPELINE_ERROR)


# ------------------------------------------------------------------ score


@app.command()
def score(
    gold_dir: Path = typer.Option(
        ..., "--gold", "-g", exists=True, file_okay=False, help="Gold annotation directory."
    ),
    pred_dir: Path = typer.Option(
        ...,
        "--pred",
        "-p",
        exists=True,
        file_okay=False,
        help="Directory written by `detect --out`.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print SpanMetrics JSON."),
    check_offsets: bool = typer.Option(
        True, "--check-offsets/--no-check-offsets", help="Verify gold offsets match the gold text."
    ),
    show_errors: bool = typer.Option(
        False, "--errors", help="List missed gold spans (leaks), then false positives."
    ),
    report: Path | None = typer.Option(
        None, "--report", help="Write the markdown metrics report to this path."
    ),
    report_model: str = typer.Option(
        "<model>", "--model", "-m", help="Model name for the report header."
    ),
) -> None:
    """Score predictions against the gold set (precision, recall, F1)."""
    try:
        gold = load_gold(gold_dir, check_offsets=check_offsets)
        pred = load_predictions(pred_dir)
    except PipelineError as exc:
        render.render_error(err, exc)
        raise typer.Exit(EXIT_PIPELINE_ERROR) from None
    if not gold:
        err.print(f"[bold red]error[/] no gold .json files in {gold_dir}")
        raise typer.Exit(2)

    gold_ids, pred_ids = {g.doc_id for g in gold}, {p.doc_id for p in pred}
    if missing := sorted(gold_ids - pred_ids):
        err.print(
            f"[yellow]warning[/] {len(missing)} gold documents have no prediction: "
            + ", ".join(missing)
        )
    if extra := sorted(pred_ids - gold_ids):
        err.print(
            f"[yellow]warning[/] {len(extra)} predictions have no gold document: {', '.join(extra)}"
        )

    try:
        metrics = scoring.score(pred, gold)
    except NotImplementedError as exc:
        raise _not_implemented(exc) from None
    if as_json:
        sys.stdout.write(metrics.model_dump_json(indent=2) + "\n")
    else:
        render.render_metrics(out, metrics, documents=len(gold))

    if show_errors or report is not None:
        pred_by_id = {p.doc_id: p for p in pred}
        matches = [scoring.match_document(pred_by_id.get(g.doc_id), g) for g in gold]
        if show_errors:
            names = {
                json.loads(f.read_text(encoding="utf-8")).get("doc_id"): f.stem
                for f in sorted(gold_dir.glob("*.json"))
            }
            out.print(scoring.format_errors(matches, names), markup=False, highlight=False)
        if report is not None:
            report.write_text(
                scoring.build_report(metrics, matches, len(gold), model=report_model),
                encoding="utf-8",
            )
            err.print(f"[dim]report written to {report}[/]")


# ------------------------------------------------------------------ health / serve


@app.command()
def health(model: str | None = typer.Option(None, "--model", "-m")) -> None:
    """Check that Ollama is reachable and the model is pulled."""
    settings = get_settings()

    async def check() -> dict:
        async with OllamaClient.from_settings(settings, model=model) as client:
            return await client.health()

    status = asyncio.run(check())
    report = {
        "status": "ok" if status["model_available"] else "degraded",
        "ollama_reachable": status["reachable"],
        "model": model or settings.model,
        "pipeline_version": PIPELINE_VERSION,
    }
    if status["detail"]:
        report["detail"] = status["detail"]
    out.print_json(json.dumps(report))
    if not status["model_available"]:
        raise typer.Exit(EXIT_PIPELINE_ERROR)


@app.command()
def build(
    input_path: Path = typer.Option(
        ..., "--input", "-i", exists=True, help="Corpus directory or file (e.g. data/corpus/docs)."
    ),
    store_path: Path | None = typer.Option(
        None, "--store", help="SQLite store to fill (default: settings.store_path)."
    ),
    out_dir: Path | None = typer.Option(
        None, "--out", "-o", help="Also write <doc_id>.json per document here."
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Ollama model."),
    mock: bool = typer.Option(
        False, "--mock", help="Deterministic mock sanitisation (no model needed)."
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Skip documents already in the store."
    ),
    index: bool = typer.Option(
        True, "--index/--no-index", help="Also embed the originals for hybrid retrieval."
    ),
) -> None:
    """Sanitise a corpus (Agents 1-3) into the document store, then build the search index."""
    from sanitise.agents.pipeline import sanitise
    from sanitise.build import BuildReport, build_store
    from sanitise.mock_mode import mock_llm, mock_sanitise
    from sanitise.rag.dense import build_retriever
    from sanitise.rag.store import SqliteDocumentStore

    settings = get_settings()
    store = SqliteDocumentStore(store_path or settings.store_path)
    docs = list(iter_documents(input_path, settings=settings, skip_too_long=True))
    err.print(f"[dim]{len(docs)} documents in {input_path}; store {store.path}[/]")

    def progress(i: int, doc: Document, status: str) -> None:
        colour = {"ok": "green", "skipped": "dim"}.get(status, "red")
        err.print(f"[{colour}]{i:>4}/{len(docs)}  {doc.doc_id}  {status}[/]")

    async def run() -> BuildReport:
        llm = mock_llm() if mock else build_llm_client(settings, model=model)
        try:
            return await build_store(
                docs,
                store,
                llm,
                out_dir=out_dir,
                resume=resume,
                sanitiser=mock_sanitise if mock else sanitise,
                progress=progress,
            )
        finally:
            await close_llm_client(llm)

    report = asyncio.run(run())
    err.print(report.summary())
    if index and settings.retrieval_mode == "hybrid":
        build_retriever(
            store,
            mode="hybrid",
            index_dir=settings.index_dir,
            model_name=settings.embedding_model,
        )
    store.close()
    if report.failed:
        raise typer.Exit(EXIT_PIPELINE_ERROR)


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    reload: bool = typer.Option(
        False, "--reload", help="Auto-reload on code changes (development)."
    ),
) -> None:
    """Run the HTTP API (FastAPI + Uvicorn)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "sanitise.api.app:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    app()
