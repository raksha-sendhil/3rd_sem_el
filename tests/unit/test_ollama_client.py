"""OllamaClient against a fake Ollama server (httpx.MockTransport) - no model needed."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from sanitise.config import Settings
from sanitise.errors import LLMTimeout, LLMUnreachable, SchemaValidationFailed
from sanitise.llm import LLMClient, OllamaClient
from sanitise.schemas import QuasiList

GOOD = json.dumps(
    {"quasi_identifiers": [{"text": "x", "category": "QUASI_COMBO", "rationale": "r"}]}
)


def make_client(handler: Callable[[httpx.Request], httpx.Response], **kwargs) -> OllamaClient:
    http = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return OllamaClient(
        model="llama3:8b", base_url="http://ollama.test", http=http, transport_retries=0, **kwargs
    )


def chat_reply(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "llama3:8b",
            "message": {"role": "assistant", "content": content},
            "done": True,
        },
    )


def test_satisfies_protocol() -> None:
    assert isinstance(OllamaClient(), LLMClient)


async def test_sends_schema_as_format_and_parses_reply() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        seen.append(json.loads(request.content))
        return chat_reply(GOOD)

    result = await make_client(handler).json_call("SYS", "USER", QuasiList)
    assert result.quasi_identifiers[0].text == "x"
    body = seen[0]
    assert body["model"] == "llama3:8b" and body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert body["format"]["properties"]["quasi_identifiers"]["type"] == "array"
    assert "$defs" not in body["format"]
    assert body["options"]["temperature"] == 0.0
    assert "seed" not in body["options"]


async def test_seed_and_num_predict_from_settings() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return chat_reply(GOOD)

    http = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    client = OllamaClient.from_settings(Settings(llm_seed=7, llm_num_predict=256), http=http)
    await client.json_call("s", "u", QuasiList)
    assert seen[0]["options"]["seed"] == 7 and seen[0]["options"]["num_predict"] == 256


async def test_repairs_invalid_output() -> None:
    replies = iter(['{"quasi_identifiers": [{"text": "x"}]}', GOOD])
    temps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        temps.append(json.loads(request.content)["options"]["temperature"])
        return chat_reply(next(replies))

    await make_client(handler).json_call("s", "u", QuasiList)
    assert temps == pytest.approx([0.0, 0.2])


async def test_exhausted_retries() -> None:
    client = make_client(lambda r: chat_reply("{}"), retries=1)
    with pytest.raises(SchemaValidationFailed) as info:
        await client.json_call("s", "u", QuasiList)
    assert info.value.attempts == 2


async def test_timeout_maps_to_llm_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMTimeout):
        await make_client(handler).json_call("s", "u", QuasiList)


async def test_connection_refused_maps_to_unreachable() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMUnreachable):
        await make_client(handler).json_call("s", "u", QuasiList)
    assert calls == 1


async def test_transport_retry_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sanitise.llm.ollama.asyncio.sleep", _no_sleep)
    attempts = iter([httpx.ConnectError("refused"), None])

    def handler(request: httpx.Request) -> httpx.Response:
        exc = next(attempts)
        if exc:
            raise exc
        return chat_reply(GOOD)

    http = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    client = OllamaClient(base_url="http://ollama.test", http=http, transport_retries=1)
    assert (await client.json_call("s", "u", QuasiList)).quasi_identifiers


async def _no_sleep(_: float) -> None:
    return None


async def test_missing_model_404() -> None:
    client = make_client(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(LLMUnreachable, match="ollama pull llama3:8b"):
        await client.json_call("s", "u", QuasiList)


async def test_health() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200, json={"models": [{"name": "llama3:8b"}, {"name": "qwen2.5:7b-instruct"}]}
        )

    assert (await make_client(handler).health())["model_available"] is True
    missing = await make_client(handler).with_options(model="mistral").health()
    assert missing["reachable"] is True and missing["model_available"] is False


async def test_health_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    status = await make_client(handler).health()
    assert status == {"reachable": False, "model_available": False, "detail": status["detail"]}


def test_from_settings() -> None:
    settings = Settings(model="qwen2.5:7b-instruct", llm_retries=2, llm_temperature_step=0.3)
    client = OllamaClient.from_settings(settings)
    assert (client.model, client.default_retries, client.temperature_step) == (
        "qwen2.5:7b-instruct",
        2,
        0.3,
    )
    clone = client.with_options(model="llama3:8b")
    assert clone.model == "llama3:8b" and client.model == "qwen2.5:7b-instruct"
    assert clone._http is client._http and clone._owns_http is False
