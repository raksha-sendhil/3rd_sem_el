"""LLM clients. Depend on ``LLMClient``; construct with ``build_llm_client``."""

from sanitise.llm.base import (
    BaseJSONClient,
    CallStats,
    LLMClient,
    default_repair_instruction,
    format_validation_error,
    inline_refs,
    parse_json_lenient,
    track_calls,
)
from sanitise.llm.factory import build_llm_client, close_llm_client
from sanitise.llm.mock import MockCall, MockLLM, fabricate
from sanitise.llm.ollama import OllamaClient

__all__ = [
    "BaseJSONClient",
    "CallStats",
    "LLMClient",
    "MockCall",
    "MockLLM",
    "OllamaClient",
    "build_llm_client",
    "close_llm_client",
    "default_repair_instruction",
    "fabricate",
    "format_validation_error",
    "inline_refs",
    "parse_json_lenient",
    "track_calls",
]
