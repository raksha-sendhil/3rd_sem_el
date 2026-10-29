"""Build the configured ``LLMClient``. The CLI and the HTTP API both go through here."""

from __future__ import annotations

import asyncio
from pathlib import Path

from sanitise.config import Settings, get_settings
from sanitise.llm.base import BaseJSONClient, LLMClient
from sanitise.llm.gemini import GeminiClient
from sanitise.llm.mock import MockLLM
from sanitise.llm.ollama import OllamaClient


def build_llm_client(
    settings: Settings,
    *,
    backend: str | None = None,
    model: str | None = None,
    mock_responses: str | Path | None = None,
) -> BaseJSONClient:
    """``backend``/``model`` override the settings; ``mock_responses`` is a MockLLM script file."""
    backend = backend or settings.llm_backend
    retry_kwargs = {
        "retries": settings.llm_retries,
        "temperature": settings.llm_temperature,
        "temperature_step": settings.llm_temperature_step,
        "temperature_max": settings.llm_temperature_max,
    }
    if backend == "mock":
        if mock_responses:
            return MockLLM.from_file(mock_responses, model=model or "mock-llm", **retry_kwargs)
        return MockLLM(model=model or "mock-llm", **retry_kwargs)
    if backend == "ollama":
        return OllamaClient.from_settings(settings, model=model)
    raise ValueError(f"unknown LLM backend {backend!r} (expected 'ollama' or 'mock')")


_gemini: dict[tuple[object, ...], GeminiClient] = {}


def agents_llm(llm: LLMClient) -> LLMClient:
    """The client Agents 2 and 3 call, when ``llm`` is the real (Ollama) client:

    * ``SANITISE_AGENTS_BACKEND=gemini``: a Gemini client for ``SANITISE_AGENTS_MODEL``, one per
      event loop and settings, so its key rotation is kept between calls;
    * otherwise the agents model and thinking switch on Ollama (``OllamaClient.for_agents``).

    A mock client is returned unchanged, so mock mode and the tests never call an API.
    """
    if not isinstance(llm, OllamaClient):
        return llm
    settings = get_settings()
    if settings.agents_backend != "gemini":
        return llm.for_agents(settings)
    key = (
        id(asyncio.get_running_loop()),
        settings.gemini_api_key,
        settings.agents_model,
        settings.gemini_thinking_level,
    )
    if key not in _gemini:
        _gemini[key] = GeminiClient.from_settings(settings)
    return _gemini[key]


def detector_llm(llm: LLMClient) -> LLMClient:
    """The client Agent 1's LLM pass calls: the Gemini client of Agents 2 and 3 when they run
    on Gemini (``SANITISE_GEMINI_FOR_AGENT1``, on by default), ``llm`` otherwise."""
    settings = get_settings()
    if settings.agents_backend == "gemini" and settings.gemini_for_agent1:
        return agents_llm(llm)
    return llm


async def close_llm_client(client: object) -> None:
    aclose = getattr(client, "aclose", None)
    if aclose is not None:
        await aclose()
