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


def build_answer_llm(settings: Settings, agents: BaseJSONClient) -> BaseJSONClient:
    """The answering model (legal build spec: Gemini on cleaned text only). ``answer_backend``
    "same" reuses the agents' model."""
    backend = settings.answer_backend
    if backend == "same":
        return agents
    if backend == "gemini":
        from sanitise.llm.gemini import GeminiClient

        return GeminiClient.from_settings(settings)
    return build_llm_client(settings, backend=backend)


def build_agent1_llm(settings: Settings, agents: BaseJSONClient) -> BaseJSONClient:
    """Agent 1's LLM pass: the agents' model, or Gemma 4 + the LoRA adapter (Part 3)."""
    if settings.agent1_backend == "transformers":
        from sanitise.llm.hf import TransformersClient

        return TransformersClient.from_settings(settings)
    return agents


async def close_llm_client(client: object) -> None:
    aclose = getattr(client, "aclose", None)
    if aclose is not None:
        await aclose()
