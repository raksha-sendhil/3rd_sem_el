"""Build the configured ``LLMClient``. The CLI and the HTTP API both go through here."""

from __future__ import annotations

from pathlib import Path

from sanitise.config import Settings
from sanitise.llm.base import BaseJSONClient
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


async def close_llm_client(client: object) -> None:
    aclose = getattr(client, "aclose", None)
    if aclose is not None:
        await aclose()
