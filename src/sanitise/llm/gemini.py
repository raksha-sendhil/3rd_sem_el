"""Gemini through the ``google-genai`` SDK.  OWNER: Person C.

Legal build spec, "Model choice": Gemini 3.5 Flash (or 3.8 Flash, one config value) answers
from cleaned text only, and may generate fictional training documents. It never sees raw
client text, and it cannot be fine-tuned, only prompted.

Fallbacks from the spec's risk table ("Gemini rate limits"), plus key rotation:

* every reply is cached on disk, keyed by the model, the messages and the generation
  settings, so a re-run after a rate limit or a notebook restart costs nothing;
* calls are spaced to at most ``rpm`` per minute;
* several API keys can be given (``KeyPool``). When a key is rate-limited (HTTP 429,
  RESOURCE_EXHAUSTED) it rests for ``cooldown_s``; when it reports a daily quota, or uses
  up its own ``daily_token_budget``, it rests until the next day; the call then moves to the
  next key. Only when every key is resting does the call fail. Token use per key and per day
  is kept in ``<cache_dir>/usage.json``, under a short fingerprint of the key, never the key.

Keys come from ``api_key`` / ``api_keys``, else the environment or Kaggle secrets:
``GEMINI_API_KEYS`` (comma-separated), ``GEMINI_API_KEY``, ``GEMINI_API_KEY_2`` ..
``GEMINI_API_KEY_9``. They are never written anywhere. ``google-genai`` is imported only when a
request is actually sent.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sanitise.config import Settings
from sanitise.errors import LLMTimeout, LLMUnreachable
from sanitise.llm.base import BaseJSONClient, Message

log = logging.getLogger(__name__)

KEY_NAMES = ("GEMINI_API_KEY", *(f"GEMINI_API_KEY_{i}" for i in range(2, 10)))
_RATE_LIMITED = re.compile(r"\b429\b|RESOURCE_EXHAUSTED|rate.?limit|quota", re.IGNORECASE)
_DAILY = re.compile(r"per.?day|PerDay|daily", re.IGNORECASE)
DAY_S = 24 * 3600.0


def _kaggle_secret(name: str) -> str | None:
    try:  # inside a Kaggle notebook
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        return UserSecretsClient().get_secret(name)
    except Exception:  # not on Kaggle, or no secret of that name is attached
        return None


def gemini_api_keys(*explicit: str | None) -> list[str]:
    """Every configured key, in order, without duplicates."""
    found: list[str] = []
    for value in (*explicit, os.environ.get("GEMINI_API_KEYS")):
        found += [k.strip() for k in (value or "").split(",") if k.strip()]
    for name in KEY_NAMES:
        if value := os.environ.get(name) or _kaggle_secret(name):
            found.append(value.strip())
    return list(dict.fromkeys(found))


def gemini_api_key(explicit: str | None = None) -> str | None:
    keys = gemini_api_keys(explicit)
    return keys[0] if keys else None


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:8]


@dataclass
class _Key:
    key: str
    rest_until: float = 0.0
    sdk: Any = None

    @property
    def id(self) -> str:
        return fingerprint(self.key)


@dataclass
class KeyPool:
    """Round-robin over keys, skipping keys that are resting or over their token budget."""

    keys: list[str]
    cooldown_s: float = 60.0
    daily_token_budget: int = 0  # per key; 0 = none
    usage_path: Path | None = None
    clock: Callable[[], float] = time.time
    _keys: list[_Key] = field(init=False)
    _next: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._keys = [_Key(k) for k in self.keys]

    def __len__(self) -> int:
        return len(self._keys)

    # usage: {"2026-11-04": {"<fingerprint>": tokens}}
    def _today(self) -> str:
        return dt.datetime.fromtimestamp(self.clock(), dt.UTC).date().isoformat()

    def _usage(self) -> dict[str, dict[str, int]]:
        if self.usage_path and self.usage_path.is_file():
            return json.loads(self.usage_path.read_text(encoding="utf-8"))
        return {}

    def used_today(self, key_id: str) -> int:
        return self._usage().get(self._today(), {}).get(key_id, 0)

    def record(self, entry: _Key, tokens: int) -> None:
        if not self.usage_path or tokens <= 0:
            return
        usage = self._usage()
        day = usage.setdefault(self._today(), {})
        day[entry.id] = day.get(entry.id, 0) + tokens
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        self.usage_path.write_text(json.dumps(usage, indent=1), encoding="utf-8")

    def _available(self, entry: _Key) -> bool:
        if entry.rest_until > self.clock():
            return False
        return not (
            self.daily_token_budget and self.used_today(entry.id) >= self.daily_token_budget
        )

    def acquire(self) -> _Key:
        """The next usable key; LLMUnreachable when all are resting or over budget."""
        if not self._keys:
            raise LLMUnreachable(
                "no Gemini API key: set GEMINI_API_KEY (or GEMINI_API_KEYS, GEMINI_API_KEY_2, "
                "...) or add them as Kaggle secrets"
            )
        for i in range(len(self._keys)):
            entry = self._keys[(self._next + i) % len(self._keys)]
            if self._available(entry):
                self._next = (self._next + i + 1) % len(self._keys)
                return entry
        soonest = min(k.rest_until for k in self._keys) - self.clock()
        raise LLMUnreachable(
            f"all {len(self._keys)} Gemini keys are rate-limited or over their daily budget"
            + (f"; the first is free again in {soonest:.0f} s" if soonest > 0 else "")
        )

    def rest(self, entry: _Key, message: str) -> None:
        daily = bool(_DAILY.search(message))
        entry.rest_until = self.clock() + (DAY_S if daily else self.cooldown_s)
        log.warning(
            "Gemini key %s hit its %s; switching key",
            entry.id,
            "daily quota" if daily else "rate limit",
        )


class GeminiClient(BaseJSONClient):
    def __init__(
        self,
        *,
        model: str = "gemini-3.5-flash",
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        cache_dir: str | Path | None = "data/cache/gemini",
        rpm: int = 10,
        timeout_s: float = 120.0,
        cooldown_s: float = 60.0,
        daily_token_budget: int = 0,
        sdk_client: Any = None,
        sdk_factory: Callable[[str], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        keys = list(api_keys or []) or gemini_api_keys(api_key)
        if sdk_client is not None and not keys:
            keys = ["injected"]  # tests: one fake SDK, no real key
        self.pool = KeyPool(
            keys,
            cooldown_s=cooldown_s,
            daily_token_budget=daily_token_budget,
            usage_path=self.cache_dir / "usage.json" if self.cache_dir else None,
        )
        self._sdk_client = sdk_client
        self._sdk_factory = sdk_factory
        self.min_interval = 60.0 / max(rpm, 1)
        self.timeout_s = timeout_s
        self._lock = asyncio.Lock()
        self._last_call = 0.0
        self.cache_hits = 0

    @classmethod
    def from_settings(cls, settings: Settings, *, model: str | None = None) -> GeminiClient:
        configured = [settings.gemini_api_key, settings.gemini_api_keys]
        return cls(
            model=model or settings.gemini_model,
            api_keys=gemini_api_keys(*configured),
            cache_dir=settings.gemini_cache_dir,
            rpm=settings.gemini_rpm,
            timeout_s=settings.llm_timeout_s,
            cooldown_s=settings.gemini_key_cooldown_s,
            daily_token_budget=settings.gemini_daily_token_budget,
            retries=settings.llm_retries,
            temperature=settings.llm_temperature,
            temperature_step=settings.llm_temperature_step,
            temperature_max=settings.llm_temperature_max,
        )

    # ---------------------------------------------------------------- cache

    def _key(self, messages: list[Message], json_schema: dict | None, *settings: Any) -> str:
        blob = json.dumps([self.model, messages, json_schema, settings], sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cached(self, key: str) -> str | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if path.is_file():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        return None

    def _store(self, key: str, text: str) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps({"model": self.model, "text": text}, ensure_ascii=False), encoding="utf-8"
        )

    # ---------------------------------------------------------------- transport

    def _sdk(self, entry: _Key) -> Any:
        if entry.sdk is None:
            if self._sdk_factory is not None:
                entry.sdk = self._sdk_factory(entry.key)
            elif self._sdk_client is not None:
                entry.sdk = self._sdk_client
            else:
                from google import genai

                entry.sdk = genai.Client(api_key=entry.key)
        return entry.sdk

    async def _throttle(self) -> None:
        async with self._lock:
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    async def _generate(
        self,
        messages: list[Message],
        json_schema: dict[str, Any] | None,
        temperature: float,
        *,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        cache_key = self._key(messages, json_schema, temperature, top_p, max_tokens)
        if (hit := self._cached(cache_key)) is not None:
            return hit
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        config: dict[str, Any] = {"temperature": temperature}
        if system:
            config["system_instruction"] = system
        if top_p is not None:
            config["top_p"] = top_p
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if json_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = json_schema

        for _ in range(max(len(self.pool), 1)):
            entry = self.pool.acquire()
            await self._throttle()
            try:
                response = await asyncio.wait_for(
                    self._sdk(entry).aio.models.generate_content(
                        model=self.model, contents=contents, config=config
                    ),
                    timeout=self.timeout_s,
                )
            except TimeoutError:
                raise LLMTimeout(
                    f"{self.model} did not answer within {self.timeout_s:.0f}s"
                ) from None
            except Exception as exc:  # the SDK raises its own error types
                if _RATE_LIMITED.search(str(exc)):
                    self.pool.rest(entry, str(exc))
                    continue  # the next key
                raise LLMUnreachable(f"Gemini call failed: {exc}") from None
            usage = getattr(response, "usage_metadata", None)
            self.pool.record(entry, int(getattr(usage, "total_token_count", 0) or 0))
            text = getattr(response, "text", None) or ""
            self._store(cache_key, text)
            return text
        self.pool.acquire()  # every key just rested: raises with the reason
        raise AssertionError("unreachable")  # pragma: no cover
