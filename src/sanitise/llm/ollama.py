"""Async Ollama client with JSON-schema-constrained decoding.

Uses ``POST /api/chat`` with ``format`` set to the target Pydantic model's JSON
schema, so Ollama constrains generation to that grammar. Validation and the
retry/repair loop come from ``BaseJSONClient``; this module only moves bytes and
maps transport failures to the spec's error codes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from sanitise.config import Settings
from sanitise.errors import LLMTimeout, LLMUnreachable
from sanitise.llm.base import BaseJSONClient, Message, OutputParser, RepairBuilder

log = logging.getLogger(__name__)


class OllamaClient(BaseJSONClient):
    def __init__(
        self,
        *,
        model: str = "llama3:8b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120.0,
        num_ctx: int = 4096,
        seed: int | None = None,
        num_predict: int | None = None,
        retries: int = 3,
        transport_retries: int = 1,
        temperature: float = 0.0,
        temperature_step: float = 0.2,
        temperature_max: float = 1.0,
        http: httpx.AsyncClient | None = None,
        **hooks: Any,
    ) -> None:
        super().__init__(
            model=model,
            retries=retries,
            temperature=temperature,
            temperature_step=temperature_step,
            temperature_max=temperature_max,
            **hooks,
        )
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx
        self.seed = seed
        self.num_predict = num_predict
        self.transport_retries = transport_retries
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        model: str | None = None,
        parser: OutputParser | None = None,
        repair_builder: RepairBuilder | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> OllamaClient:
        hooks: dict[str, Any] = {}
        if parser is not None:
            hooks["parser"] = parser
        if repair_builder is not None:
            hooks["repair_builder"] = repair_builder
        return cls(
            model=model or settings.model,
            base_url=settings.ollama_base_url,
            timeout_s=settings.llm_timeout_s,
            num_ctx=settings.llm_num_ctx,
            seed=settings.llm_seed,
            num_predict=settings.llm_num_predict,
            retries=settings.llm_retries,
            transport_retries=settings.transport_retries,
            temperature=settings.llm_temperature,
            temperature_step=settings.llm_temperature_step,
            temperature_max=settings.llm_temperature_max,
            http=http,
            **hooks,
        )

    def with_options(self, *, model: str | None = None, retries: int | None = None) -> OllamaClient:
        """A copy with a different model or retry budget that shares this client's HTTP pool."""
        clone = super().with_options(model=model, retries=retries)
        clone._owns_http = False  # closing the copy must not close the shared pool
        return clone

    # ---------------------------------------------------------------- transport

    async def _generate(
        self, messages: list[Message], json_schema: dict[str, Any], temperature: float
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": json_schema,
            "options": self._options(temperature),
        }
        data = await self._post("/api/chat", payload)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError):
            raise LLMUnreachable(
                f"unexpected response shape from Ollama: {str(data)[:200]}"
            ) from None

    def _options(self, temperature: float) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature, "num_ctx": self.num_ctx}
        if self.seed is not None:
            options["seed"] = self.seed
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        return options

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.transport_retries + 1):
            try:
                response = await self._http.post(path, json=payload, timeout=self.timeout_s)
            except httpx.TimeoutException:
                raise LLMTimeout(
                    f"{self.model} did not answer within {self.timeout_s:.0f}s at {self.base_url}"
                ) from None
            except httpx.TransportError as exc:
                if attempt < self.transport_retries:
                    delay = 0.5 * 2**attempt
                    log.warning("Ollama unreachable (%s); retrying in %.1fs", exc, delay)
                    await asyncio.sleep(delay)
                    continue
                raise LLMUnreachable(f"cannot reach Ollama at {self.base_url}: {exc}") from None
            if response.status_code == 404:
                raise LLMUnreachable(
                    f"Ollama returned 404 for model {self.model!r} - run `ollama pull {self.model}`"
                )
            if response.status_code >= 400:
                raise LLMUnreachable(
                    f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
                )
            try:
                return response.json()
            except ValueError:
                raise LLMUnreachable(
                    f"Ollama sent a non-JSON body: {response.text[:200]}"
                ) from None
        raise AssertionError("unreachable")  # pragma: no cover

    # ---------------------------------------------------------------- health

    async def list_models(self) -> list[str]:
        try:
            response = await self._http.get("/api/tags", timeout=min(self.timeout_s, 5.0))
            response.raise_for_status()
        except httpx.TimeoutException:
            raise LLMTimeout(f"Ollama at {self.base_url} did not answer /api/tags") from None
        except httpx.HTTPError as exc:
            raise LLMUnreachable(f"cannot reach Ollama at {self.base_url}: {exc}") from None
        return [m.get("name", "") for m in response.json().get("models", [])]

    async def health(self) -> dict[str, Any]:
        """``{"reachable": bool, "model_available": bool, "detail": str | None}``."""
        try:
            names = await self.list_models()
        except (LLMTimeout, LLMUnreachable) as exc:
            return {"reachable": False, "model_available": False, "detail": exc.detail}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        available = wanted in names or self.model in names
        detail = None if available else f"model {self.model!r} not pulled; have {names}"
        return {"reachable": True, "model_available": available, "detail": detail}

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
