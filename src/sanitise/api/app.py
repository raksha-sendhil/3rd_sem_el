"""FastAPI application (spec 8.4).

    POST /v1/detect        { doc_id, text, options }  -> DetectorResult
    POST /v1/detect/rules  { doc_id, text }           -> DetectorOutput   (rule pass only)
    POST /v1/detect/llm    { doc_id, text, explicit } -> DetectorOutput   (LLM pass only)
    POST /v1/score         { pred[], gold[] }         -> SpanMetrics
    GET  /v1/health   -> { status, ollama_reachable, model, pipeline_version }

Every ``PipelineError`` becomes the spec 8.5 error envelope with an HTTP status
from the error class (413 DOC_TOO_LONG, 502 SCHEMA_VALIDATION_FAILED, 503
LLM_UNREACHABLE, 504 LLM_TIMEOUT, 500 OFFSET_OUT_OF_RANGE). Malformed request
bodies get FastAPI's standard 422. A stage that is still a placeholder answers 501.

Phase 3a (docs/phase3_plan.md 6.4) adds sanitise, documents, sessions, query, compare and
attacks. With ``SANITISE_LLM_BACKEND=mock`` those run on the 20 fixtures and the sample
people and attacks (``sanitise.mock_mode``), so the dashboard works without a model.

Run with ``sanitise serve`` or ``uvicorn sanitise.api.app:create_app --factory``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from sanitise.agents.detector.pipeline import detect, detect_llm, detect_rules, detect_with_trace
from sanitise.agents.extractor.pipeline import extract
from sanitise.agents.pipeline import sanitise
from sanitise.agents.rewriter.pipeline import rewrite
from sanitise.api.models import (
    AttackRunRequest,
    CompareRequest,
    DetectLLMRequest,
    DetectRequest,
    DetectRulesRequest,
    DocumentSummary,
    DocumentView,
    ExtractRequest,
    HealthResponse,
    QueryRequest,
    RewriteRequest,
    SanitiseRequest,
    ScoreRequest,
    SessionCreated,
)
from sanitise.attacks.runner import (
    get_attack,
    iter_attack,
    load_attacks,
    load_people,
    run_attack,
    summarise,
)
from sanitise.config import Settings, get_settings
from sanitise.errors import PipelineError
from sanitise.evaluation import scoring
from sanitise.llm.base import BaseJSONClient
from sanitise.llm.factory import build_llm_client, close_llm_client
from sanitise.llm.ollama import OllamaClient
from sanitise.loader import compute_doc_id
from sanitise.mock_mode import (
    SAMPLE_DIR,
    build_mock_context,
    mock_facts,
    mock_llm,
    mock_rewrite,
    mock_sanitise,
)
from sanitise.rag.answer import RagContext, answer, compare
from sanitise.rag.dense import build_retriever
from sanitise.rag.store import SqliteDocumentStore
from sanitise.schemas import (
    PIPELINE_VERSION,
    Action,
    AttackDefinition,
    AttackRun,
    CompareResponse,
    DetectorOutput,
    DetectorResult,
    ExtractorOutput,
    RagAnswer,
    RewriterOutput,
    SanitisedRecord,
    SessionTrace,
    SpanMetrics,
)


def create_app(
    settings: Settings | None = None,
    llm: BaseJSONClient | None = None,
    rag: RagContext | None = None,
) -> FastAPI:
    """Build the app. Tests pass ``llm=MockLLM(...)`` and may pass a ready ``rag`` context;
    otherwise both come from settings (mock mode builds the fixture corpus on first use)."""
    settings = settings or get_settings()
    mock = settings.llm_backend == "mock"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = llm is None
        if llm is not None:
            app.state.llm = llm
        else:
            app.state.llm = mock_llm() if mock else build_llm_client(settings)
        try:
            yield
        finally:
            if owned:
                await close_llm_client(app.state.llm)

    app = FastAPI(
        title="sanitise - Agent 1: Sensitive Information Detector",
        version=PIPELINE_VERSION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.rag = rag
    rag_lock = asyncio.Lock()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(PipelineError)
    async def _pipeline_error(_: Request, exc: PipelineError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status, content=exc.to_envelope().model_dump(mode="json")
        )

    @app.exception_handler(NotImplementedError)
    async def _not_implemented(_: Request, exc: NotImplementedError) -> JSONResponse:
        return JSONResponse(status_code=501, content={"detail": str(exc) or "not implemented"})

    def client_for(
        request: Request, model: str | None = None, retries: int | None = None
    ) -> BaseJSONClient:
        base: BaseJSONClient = request.app.state.llm
        if model is None and retries is None:
            return base
        return base.with_options(model=model, retries=retries)

    @app.post("/v1/detect", response_model=DetectorResult)
    async def post_detect(body: DetectRequest, request: Request) -> DetectorResult:
        llm = client_for(request, body.options.model, body.options.retries)
        result, _ = await detect_with_trace(
            body.doc_id or compute_doc_id(body.text),
            body.text,
            llm,
            run_llm=body.options.llm,
            max_chars=settings.max_doc_chars,
        )
        return result

    @app.post("/v1/detect/rules", response_model=DetectorOutput)
    async def post_detect_rules(body: DetectRulesRequest) -> DetectorOutput:
        return await detect_rules(
            body.doc_id or compute_doc_id(body.text), body.text, max_chars=settings.max_doc_chars
        )

    @app.post("/v1/detect/llm", response_model=DetectorOutput)
    async def post_detect_llm(body: DetectLLMRequest, request: Request) -> DetectorOutput:
        return await detect_llm(
            body.doc_id or compute_doc_id(body.text),
            body.text,
            body.explicit,
            client_for(request),
            max_chars=settings.max_doc_chars,
        )

    @app.post("/v1/score", response_model=SpanMetrics)
    async def post_score(body: ScoreRequest) -> SpanMetrics:
        return scoring.score(body.pred, body.gold)

    # ---------------------------------------------------------- Phase 3a (plan 6.4)

    async def rag_for(request: Request) -> RagContext:
        """The store/retriever/sessions context, built on first use."""
        if request.app.state.rag is None:
            async with rag_lock:
                if request.app.state.rag is None:
                    if mock:
                        request.app.state.rag = await build_mock_context(request.app.state.llm)
                    else:
                        store = SqliteDocumentStore(settings.store_path)
                        retriever = await asyncio.to_thread(
                            build_retriever,
                            store,
                            mode=settings.retrieval_mode,
                            index_dir=settings.index_dir,
                            model_name=settings.embedding_model,
                        )
                        request.app.state.rag = RagContext(store=store, retriever=retriever)
        return request.app.state.rag

    def attack_sources() -> tuple[Path, Path]:
        if mock:
            return SAMPLE_DIR / "attacks", SAMPLE_DIR / "people.json"
        return Path(settings.attacks_dir), Path(settings.people_path)

    @app.post("/v1/sanitise", response_model=SanitisedRecord)
    async def post_sanitise(body: SanitiseRequest, request: Request) -> SanitisedRecord:
        doc_id = body.doc_id or compute_doc_id(body.text)
        run = mock_sanitise if mock else sanitise
        return await run(doc_id, body.text, client_for(request))

    @app.post("/v1/extract", response_model=ExtractorOutput)
    async def post_extract(body: ExtractRequest, request: Request) -> ExtractorOutput:
        """Agent 2 alone. It needs Agent 1's spans to find the subjects, so Agent 1 runs first."""
        doc_id = body.doc_id or compute_doc_id(body.text)
        llm = client_for(request)
        detected = await detect(doc_id, body.text, llm)
        if mock:
            return ExtractorOutput(doc_id=doc_id, facts=mock_facts(detected.output))
        _, extracted = await extract(doc_id, body.text, detected.output, llm)
        return extracted

    @app.post("/v1/rewrite", response_model=RewriterOutput)
    async def post_rewrite(body: RewriteRequest, request: Request) -> RewriterOutput:
        detected = DetectorOutput(doc_id=body.doc_id, spans=body.spans)
        extracted = ExtractorOutput(doc_id=body.doc_id, facts=body.facts)
        if mock:
            return mock_rewrite(body.doc_id, detected, extracted)
        return await rewrite(body.doc_id, detected, extracted, client_for(request))

    @app.get("/v1/documents", response_model=list[DocumentSummary])
    async def get_documents(
        request: Request, limit: int = 50, offset: int = 0
    ) -> list[DocumentSummary]:
        ctx = await rag_for(request)
        out = []
        for doc_id in ctx.store.ids()[offset : offset + limit]:
            record = ctx.store.sanitised(doc_id)
            out.append(
                DocumentSummary(
                    doc_id=doc_id,
                    preview=record.rewriter.safe_text[:120],
                    n_changes=sum(d.action is not Action.KEEP for d in record.rewriter.decisions),
                )
            )
        return out

    @app.get("/v1/documents/{doc_id}", response_model=DocumentView)
    async def get_document(doc_id: str, request: Request) -> DocumentView:
        ctx = await rag_for(request)
        return DocumentView(
            doc_id=doc_id, original=ctx.store.original(doc_id), record=ctx.store.sanitised(doc_id)
        )

    @app.post("/v1/sessions", response_model=SessionCreated)
    async def post_session(request: Request) -> SessionCreated:
        return SessionCreated(session_id=(await rag_for(request)).sessions.create())

    @app.get("/v1/sessions/{session_id}", response_model=SessionTrace)
    async def get_session(session_id: str, request: Request) -> SessionTrace:
        return (await rag_for(request)).sessions.get(session_id)

    @app.post("/v1/query", response_model=RagAnswer)
    async def post_query(body: QueryRequest, request: Request) -> RagAnswer:
        ctx = await rag_for(request)
        return await answer(
            ctx,
            body.query,
            body.pipeline,
            body.session_id,
            client_for(request),
            body.k or settings.retrieval_k,
        )

    @app.post("/v1/compare", response_model=CompareResponse)
    async def post_compare(body: CompareRequest, request: Request) -> CompareResponse:
        ctx = await rag_for(request)
        return await compare(
            ctx, body.query, body.session_id, client_for(request), body.k or settings.retrieval_k
        )

    @app.get("/v1/attacks", response_model=list[AttackDefinition])
    async def get_attacks() -> list[AttackDefinition]:
        attacks_dir, _ = attack_sources()
        return list(load_attacks(attacks_dir).values())

    @app.post("/v1/attacks/{attack_id}/run", response_model=AttackRun)
    async def post_attack_run(
        attack_id: str, request: Request, body: AttackRunRequest | None = None
    ) -> AttackRun:
        attacks_dir, people_path = attack_sources()
        attack = get_attack(load_attacks(attacks_dir), attack_id)
        ctx = await rag_for(request)
        return await run_attack(
            ctx,
            attack,
            load_people(people_path),
            client_for(request),
            body.session_id if body else None,
        )

    @app.get("/v1/attacks/{attack_id}/stream")
    async def get_attack_stream(
        attack_id: str, request: Request, session_id: str | None = None
    ) -> StreamingResponse:
        """Server-Sent Events: one ``round`` event per finished round, then ``done`` with the
        full ``AttackRun`` (or ``error`` with the error envelope)."""
        attacks_dir, people_path = attack_sources()
        attack = get_attack(load_attacks(attacks_dir), attack_id)
        ctx = await rag_for(request)
        people = load_people(people_path)
        llm = client_for(request)
        sid = session_id or ctx.sessions.create()

        def event(name: str, data: str) -> str:
            return f"event: {name}\ndata: {data}\n\n"

        async def events() -> AsyncIterator[str]:
            rounds = []
            try:
                async for round_ in iter_attack(ctx, attack, people, llm, sid):
                    rounds.append(round_)
                    yield event("round", round_.model_dump_json())
                yield event("done", summarise(attack, sid, rounds).model_dump_json())
            except PipelineError as exc:
                yield event("error", exc.to_envelope().model_dump_json())

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    @app.get("/v1/health", response_model=HealthResponse)
    async def get_health(request: Request) -> HealthResponse:
        llm = request.app.state.llm
        if isinstance(llm, OllamaClient):
            status = await llm.health()
            reachable, usable = status["reachable"], status["model_available"]
        else:
            # Mock backend: the API is usable without a model; still report whether Ollama is up.
            async with OllamaClient.from_settings(settings) as probe:
                reachable = (await probe.health())["reachable"]
            usable = True
        return HealthResponse(
            status="ok" if usable else "degraded",
            ollama_reachable=reachable,
            model=str(getattr(llm, "model", "unknown")),
            pipeline_version=PIPELINE_VERSION,
        )

    return app
