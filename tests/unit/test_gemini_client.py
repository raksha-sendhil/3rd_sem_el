"""GeminiClient against a fake Gemini API (httpx.MockTransport) - no key, no network."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from sanitise.agents.extractor import prompt as extract_prompt
from sanitise.config import Settings
from sanitise.errors import LLMUnreachable
from sanitise.llm import MockLLM
from sanitise.llm.factory import agents_llm
from sanitise.llm.gemini import GeminiClient
from sanitise.schemas import Subject


def client(handler: Callable[[httpx.Request], httpx.Response], **kwargs) -> GeminiClient:
    http = httpx.AsyncClient(
        base_url="https://gemini.test/v1beta", transport=httpx.MockTransport(handler)
    )
    kwargs.setdefault("api_keys", ["key-1"])
    return GeminiClient(min_interval_s=0.0, http=http, **kwargs)


def answer(text: str, thought: str | None = None) -> httpx.Response:
    parts = ([{"text": thought, "thought": True}] if thought else []) + [{"text": text}]
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]}
    )


def error(status: int, message: str, retry: str | None = None) -> httpx.Response:
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry}]
    body = {"error": {"code": status, "message": message, "details": details if retry else []}}
    return httpx.Response(status, json=body)


async def test_request_body_and_answer_without_the_thinking() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answer('{"facts": []}', thought="The writer is asking about...")

    gemini = client(handler)
    result = await gemini.json_call("SYSTEM", "USER", extract_prompt.LLMFactList)
    assert result.facts == []
    request = seen[0]
    body = json.loads(request.content)
    assert request.url.path.endswith("/models/gemini-3.1-flash-lite:generateContent")
    assert request.headers["x-goog-api-key"] == "key-1"
    assert body["systemInstruction"] == {"parts": [{"text": "SYSTEM"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "USER"}]}]
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"]["title"] == "FactList"
    assert config["thinkingConfig"] == {"thinkingLevel": "low"}
    assert "temperature" not in config  # the model's default


async def test_per_minute_limit_is_waited_out() -> None:
    replies = iter([error(429, "Quota exceeded: requests per minute", retry="0s"), answer("ok")])
    gemini = client(lambda _: next(replies))
    assert await gemini.text_call("s", "u") == "ok"


async def test_daily_limit_moves_to_the_next_key_then_says_so() -> None:
    keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        keys.append(request.headers["x-goog-api-key"])
        if request.headers["x-goog-api-key"] == "key-2" and len(keys) == 2:
            return answer("ok")
        return error(429, "Quota exceeded for metric: GenerateRequestsPerDayPerProjectPerModel")

    gemini = client(handler, api_keys=["key-1", "key-2"])
    assert await gemini.text_call("s", "u") == "ok"
    assert keys == ["key-1", "key-2"]
    with pytest.raises(LLMUnreachable, match="daily limit reached on all 2"):
        await gemini.text_call("s", "u")


async def test_a_rejected_schema_falls_back_to_plain_json() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if "responseJsonSchema" in bodies[-1]["generationConfig"]:
            return error(400, "Invalid JSON payload: unknown field responseJsonSchema")
        return answer('{"facts": []}')

    gemini = client(handler)
    await gemini.json_call("s", "u", extract_prompt.LLMFactList)
    await gemini.json_call("s", "u", extract_prompt.LLMFactList)
    assert len(bodies) == 3  # one rejected, then plain JSON from then on


def test_no_key_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="SANITISE_GEMINI_API_KEY"):
        GeminiClient(api_keys=[""])


def test_mock_mode_never_uses_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SANITISE_AGENTS_BACKEND", "gemini")
    monkeypatch.setenv("SANITISE_GEMINI_API_KEY", "key-1")
    mock = MockLLM()
    assert agents_llm(mock) is mock
    assert Settings().agents_backend == "gemini"


async def test_one_call_per_document_with_subjects_from_the_document() -> None:
    llm = MockLLM(
        by_schema={
            "FactList": [
                {
                    "facts": [
                        {"subject": "writer.son", "relation": "has_symptom", "value": "fever"},
                        {"subject": "writer", "relation": "other", "value": "former smoker"},
                    ]
                }
            ]
        }
    )
    llm.one_call_per_document = True  # type: ignore[attr-defined]
    subjects = [
        Subject(id="writer", label="the writer", mentions=["I", "Asha Rao"]),
        Subject(id="writer.son", label="the writer's son", mentions=["My son", "he"]),
    ]
    facts = await extract_prompt.extract_facts("text", subjects, [], llm)
    assert [(f.subject, f.relation) for f in facts] == [
        ("writer.son", "has_symptom"),
        ("writer", "other"),
    ]
    assert len(llm.calls) == 1
    assert "- writer: the writer (Asha Rao)" in llm.calls[0].user  # names, not pronouns
    subject_enum = llm.calls[0].json_schema["properties"]["facts"]["items"]["properties"]
    assert subject_enum["subject"]["enum"] == ["writer", "writer.son"]


def test_document_prompt_examples_parse_with_their_schema() -> None:
    import re

    blocks = re.findall(r"OUTPUT:\n(\{.*?\]\})", extract_prompt.DOCUMENT_PROMPT, re.S)
    assert len(blocks) == 3
    schema = extract_prompt.document_schema(["writer", "writer.wife", "writer.son", "person_1"])
    for block in blocks:
        schema.model_validate_json(block)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("sanitise.llm.gemini.asyncio.sleep", fake_sleep)
    return slept


async def test_a_busy_model_is_retried_then_the_next_model_is_used(no_sleep: list[float]) -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = request.url.path.split("/models/")[1].split(":")[0]
        models.append(model)
        if model == "gemini-a":
            return error(503, "This model is currently experiencing high demand.")
        return answer("ok")

    gemini = client(handler, model="gemini-a,gemini-b")
    assert await gemini.text_call("s", "u") == "ok"
    assert models == ["gemini-a", "gemini-a", "gemini-b"]
    assert no_sleep == [10.0]  # one wait, then the switch
    assert await gemini.text_call("s", "u") == "ok" and models[-1] == "gemini-b"  # stays


async def test_a_busy_single_model_backs_off_then_fails(no_sleep: list[float]) -> None:
    gemini = client(lambda _: error(503, "high demand"), max_server_errors=5)
    with pytest.raises(LLMUnreachable, match="503"):
        await gemini.text_call("s", "u")
    assert no_sleep == [10.0, 20.0, 40.0, 60.0, 60.0]


async def test_daily_limit_on_every_key_moves_to_the_next_model() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = request.url.path.split("/models/")[1].split(":")[0]
        seen.append((model, request.headers["x-goog-api-key"]))
        if model == "gemini-a":
            return error(429, "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel")
        return answer("ok")

    gemini = client(handler, model="gemini-a,gemini-b", api_keys=["k1", "k2"])
    assert await gemini.text_call("s", "u") == "ok"
    assert seen == [("gemini-a", "k1"), ("gemini-a", "k2"), ("gemini-b", "k1")]


async def test_agent1_llm_pass_uses_gemini_unless_switched_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sanitise.config import get_settings
    from sanitise.llm import OllamaClient
    from sanitise.llm.factory import detector_llm

    monkeypatch.setenv("SANITISE_AGENTS_BACKEND", "gemini")
    monkeypatch.setenv("SANITISE_GEMINI_API_KEY", "key-1")
    get_settings.cache_clear()
    async with OllamaClient(model="llama3:8b") as local:
        assert isinstance(detector_llm(local), GeminiClient)
        monkeypatch.setenv("SANITISE_GEMINI_FOR_AGENT1", "false")
        get_settings.cache_clear()
        assert detector_llm(local) is local


def test_case_file_prompt_examples_parse_with_the_case_schema() -> None:
    import re

    blocks = re.findall(r"OUTPUT:\n(\{.*?\]\})", extract_prompt.CASE_DOCUMENT_PROMPT, re.S)
    assert len(blocks) == 4
    schema = extract_prompt.document_schema(
        ["writer", "writer.wife", "writer.son", "person_1"], extract_prompt.CASE_RELATIONS
    )
    for block in blocks:
        schema.model_validate_json(block)
