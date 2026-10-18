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

Run with ``sanitise serve`` or ``uvicorn sanitise.api.app:create_app --factory``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sanitise.agents.detector.pipeline import detect_llm, detect_rules, detect_with_trace
from sanitise.api.models import (
    DetectLLMRequest,
    DetectRequest,
    DetectRulesRequest,
    HealthResponse,
    ScoreRequest,
)
from sanitise.config import Settings, get_settings
from sanitise.errors import PipelineError
from sanitise.evaluation import scoring
from sanitise.llm.base import BaseJSONClient
from sanitise.llm.factory import build_llm_client, close_llm_client
from sanitise.llm.ollama import OllamaClient
from sanitise.loader import compute_doc_id
from sanitise.schemas import PIPELINE_VERSION, DetectorOutput, DetectorResult, SpanMetrics


def create_app(settings: Settings | None = None, llm: BaseJSONClient | None = None) -> FastAPI:
    """Build the app. Tests pass ``llm=MockLLM(...)``; otherwise the client comes from settings."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = llm is None
        app.state.llm = llm or build_llm_client(settings)
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
