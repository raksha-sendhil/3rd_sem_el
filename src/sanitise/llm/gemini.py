"""Gemini API client (free tier) for Agents 2 and 3.

Uses ``POST /v1beta/models/{model}:generateContent`` with the system prompt as
``systemInstruction``. JSON calls ask for ``application/json`` constrained by the Pydantic
model's JSON Schema (``responseJsonSchema``); validation and the repair loop come from
``BaseJSONClient``, as for Ollama.

Free-tier limits are per Google Cloud project, per minute and per day. The client keeps to
them itself:

* requests are spaced at least ``min_interval_s`` apart (4.5 s keeps under 15 a minute), across
  every copy of the client in the process;
* a 429 that asks to wait (a per-minute limit) is waited out and retried;
* a 429 for a daily limit moves on to the next key in ``api_keys``. A new key only helps if it
  belongs to a different Google Cloud project: keys in one project share its quota.

When every key is used up the call fails with ``LLM_UNREACHABLE`` and says so.

Thinking is set with ``thinkingConfig.thinkingLevel`` ("minimal", "low", "medium", "high").
If the API rejects an optional field (the schema form or the thinking setting), the request is
sent once more without it and the client remembers that for later calls.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import httpx

from sanitise.config import Settings
from sanitise.errors import LLMTimeout, LLMUnreachable
from sanitise.llm.base import BaseJSONClient, Message

log = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# shared by every GeminiClient in the process: the per-minute limit is per project, not per copy
_pace_locks: dict[int, asyncio.Lock] = {}  # one per event loop
_last_request = 0.0


def _pace_lock() -> asyncio.Lock:
    return _pace_locks.setdefault(id(asyncio.get_running_loop()), asyncio.Lock())


class GeminiClient(BaseJSONClient):
    # Agent 2 asks a strong model for every person's facts in one call (extractor/prompt.py)
    one_call_per_document = True

    def __init__(
        self,
        *,
        api_keys: list[str],
        model: str = "gemini-3.1-flash-lite",
        thinking_level: str | None = "low",
        temperature: float | None = None,
        max_output_tokens: int = 4096,
        timeout_s: float = 120.0,
        min_interval_s: float = 4.5,
        max_waits: int = 6,
        retries: int = 3,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        keys = [k.strip() for k in api_keys if k and k.strip()]
        if not keys:
            raise ValueError("no Gemini API key: set SANITISE_GEMINI_API_KEY")
        super().__init__(model=model, retries=retries, temperature=temperature or 0.0)
        self.api_keys = keys
        self.key_index = 0
        self.thinking_level = thinking_level
        self.fixed_temperature = temperature  # None: the model's default (recommended)
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.min_interval_s = min_interval_s
        self.max_waits = max_waits
        self.use_json_schema = True
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(base_url=BASE_URL, timeout=timeout_s)

    @classmethod
    def from_settings(
        cls, settings: Settings, *, http: httpx.AsyncClient | None = None
    ) -> GeminiClient:
        return cls(
            api_keys=(settings.gemini_api_key or "").split(","),
            model=settings.agents_model or "gemini-3.1-flash-lite",
            thinking_level=settings.gemini_thinking_level,
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_output_tokens,
            timeout_s=settings.llm_timeout_s,
            min_interval_s=settings.gemini_min_interval_s,
            retries=settings.llm_retries,
            http=http,
        )

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> GeminiClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ---------------------------------------------------------------- request body

    def _body(
        self,
        messages: list[Message],
        json_schema: dict[str, Any] | None,
        temperature: float,
        top_p: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        config: dict[str, Any] = {"maxOutputTokens": max_tokens or self.max_output_tokens}
        if self.fixed_temperature is not None:
            # the repair loop's escalation applies on top of a fixed base temperature
            config["temperature"] = max(self.fixed_temperature, temperature)
        if top_p is not None:
            config["topP"] = top_p
        if json_schema is not None:
            config["responseMimeType"] = "application/json"
            if self.use_json_schema:
                config["responseJsonSchema"] = json_schema
        if self.thinking_level:
            config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        body: dict[str, Any] = {"contents": contents, "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    # ---------------------------------------------------------------- transport

    async def _generate(
        self,
        messages: list[Message],
        json_schema: dict[str, Any] | None,
        temperature: float,
        *,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        waits = 0
        while True:
            body = self._body(messages, json_schema, temperature, top_p, max_tokens)
            response = await self._post(body)
            if response.status_code == 200:
                return _text(response.json(), self.model)
            full = response.text
            detail = full[:600]
            if response.status_code == 400 and self._drop_optional(full, json_schema):
                continue
            if response.status_code == 429:
                if _daily_limit(full):
                    self._next_key(detail)
                    continue
                waits += 1
                if waits > self.max_waits:
                    raise LLMUnreachable(f"Gemini kept rate-limiting {self.model}: {detail}")
                delay = _retry_delay(full)
                log.warning("Gemini per-minute limit; waiting %.0f s", delay)
                await asyncio.sleep(delay)
                continue
            if response.status_code >= 500 and waits < self.max_waits:
                waits += 1
                await asyncio.sleep(5.0 * waits)
                continue
            raise LLMUnreachable(
                f"Gemini returned {response.status_code} for {self.model}: {detail}"
            )

    def _drop_optional(self, detail: str, json_schema: dict[str, Any] | None) -> bool:
        """After a 400: stop sending an optional field the API rejected. True to retry."""
        lower = detail.lower()
        if json_schema is not None and self.use_json_schema and "schema" in lower:
            log.warning("Gemini rejected the JSON schema; asking for plain JSON instead")
            self.use_json_schema = False
            return True
        if self.thinking_level and "think" in lower:
            log.warning("Gemini rejected thinking level %r; using the default", self.thinking_level)
            self.thinking_level = None
            return True
        return False

    def _next_key(self, detail: str) -> None:
        if self.key_index + 1 >= len(self.api_keys):
            raise LLMUnreachable(
                f"Gemini daily limit reached on all {len(self.api_keys)} API key(s). Add a key "
                f"from a different Google Cloud project, or wait until tomorrow. {detail}"
            )
        self.key_index += 1
        log.warning(
            "Gemini daily limit on key %d; switching to key %d", self.key_index, self.key_index + 1
        )

    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        global _last_request
        async with _pace_lock():
            wait = _last_request + self.min_interval_s - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            _last_request = time.monotonic()
        try:
            return await self._http.post(
                f"/models/{self.model}:generateContent",
                json=body,
                headers={"x-goog-api-key": self.api_keys[self.key_index]},
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException:
            raise LLMTimeout(f"{self.model} did not answer within {self.timeout_s:.0f}s") from None
        except httpx.TransportError as exc:
            raise LLMUnreachable(f"cannot reach the Gemini API: {exc}") from None


def _text(data: dict[str, Any], model: str) -> str:
    """The answer's text: every non-thought part of the first candidate."""
    try:
        candidate = data["candidates"][0]
    except (KeyError, IndexError, TypeError):
        reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise LLMUnreachable(f"Gemini ({model}) returned no answer: {reason}") from None
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text and candidate.get("finishReason") not in (None, "STOP"):
        raise LLMUnreachable(f"Gemini ({model}) stopped early: {candidate.get('finishReason')}")
    return text


def _daily_limit(detail: str) -> bool:
    return bool(re.search(r"per ?day|PerDay|daily", detail, re.IGNORECASE))


def _retry_delay(detail: str) -> float:
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', detail)
    return float(m.group(1)) + 1.0 if m else 30.0
