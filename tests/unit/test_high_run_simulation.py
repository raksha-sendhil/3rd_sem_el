"""A long run against a fake Gemini, for every client the eval uses: 8 models x 5 keys, each
(model, key) pair with its own daily quota (some 0, as for a model the free tier does not offer),
one model unknown to the keys (404), and random busy (503) and per-minute (429) replies.

What a high run needs, checked here because a short run never reaches a daily limit:
- every request the quotas allow is used (no model or key is crossed off too early);
- no call loops: requests per call stay bounded;
- once everything is spent, each call fails at once instead of waiting.
"""

from __future__ import annotations

import io
import random
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

MODELS = [f"m{i}" for i in range(8)]
KEYS = [f"key-{i}" for i in range(5)]
UNKNOWN = "m6"  # no key can use it (404)


class FakeGemini:
    """Quota bookkeeping shared by the three fakes. ``reply(model, key)`` is 'ok', 'daily',
    'minute', 'busy' or 'unknown'."""

    def __init__(self, seed: int, busy: float = 0.1, minute: float = 0.15) -> None:
        rng = random.Random(seed)
        self.rng = rng
        self.left = {(m, k): rng.choice([0, 1, 2, 3, 4]) for m in MODELS for k in KEYS}
        for k in KEYS:
            self.left[(UNKNOWN, k)] = 0
        self.capacity = sum(self.left.values())
        self.busy, self.minute = busy, minute
        self.requests = 0

    def reply(self, model: str, key: str) -> str:
        self.requests += 1
        if model == UNKNOWN:
            return "unknown"
        if self.rng.random() < self.busy:
            return "busy"
        if self.rng.random() < self.minute:
            return "minute"
        if self.left[(model, key)] <= 0:
            return "daily"
        self.left[(model, key)] -= 1
        return "ok"


DAILY = "Quota exceeded for metric: GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20"
MINUTE = "Quota exceeded for metric: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"


# ------------------------------------------------------------------ health and finance agents


@pytest.mark.parametrize("pkg", ["sanitise", "sanitise_finance"])
@pytest.mark.parametrize("seed", [1, 2, 3])
async def test_agents_client_uses_every_quota_and_never_loops(pkg, seed, monkeypatch) -> None:
    import importlib

    mod = importlib.import_module(f"{pkg}.llm.gemini")
    errors = importlib.import_module(f"{pkg}.errors")

    async def no_sleep(_: float) -> None:
        pass

    monkeypatch.setattr(mod.asyncio, "sleep", no_sleep)
    fake = FakeGemini(seed)

    def handler(request: httpx.Request) -> httpx.Response:
        model = request.url.path.split("/models/")[1].split(":")[0]
        kind = fake.reply(model, request.headers["x-goog-api-key"])
        if kind == "ok":
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]},
                                                             "finishReason": "STOP"}]})
        status, message = {"daily": (429, DAILY), "minute": (429, MINUTE), "busy": (503, "overloaded"),
                           "unknown": (404, "model not found")}[kind]
        details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1s"}]
        return httpx.Response(status, json={"error": {"code": status, "message": message,
                                                      "details": details}})

    http = httpx.AsyncClient(base_url="https://gemini.test/v1beta", transport=httpx.MockTransport(handler))
    client = mod.GeminiClient(api_keys=KEYS, model=",".join(MODELS), min_interval_s=0.0, http=http)
    ok = failed = 0
    per_call: list[int] = []
    for _ in range(fake.capacity * 2 + 20):
        before = fake.requests
        try:
            await client.text_call("s", "u")
            ok += 1
        except errors.LLMUnreachable:
            failed += 1
        per_call.append(fake.requests - before)
    assert ok == fake.capacity, "a model or key was crossed off while it still had quota"
    assert max(per_call) <= 4 * len(MODELS) * len(KEYS), "a call kept retrying"
    assert max(per_call[-10:]) <= 6, "with everything spent, a call should fail at once"
    await http.aclose()


# ------------------------------------------------------------------ legal agents


@pytest.mark.parametrize("seed", [1, 2, 3])
async def test_legal_client_uses_every_quota_and_never_loops(seed, tmp_path, monkeypatch) -> None:
    from sanitise_legal.errors import LLMUnreachable
    from sanitise_legal.llm import gemini as mod

    clock = {"t": 1_000_000.0}

    async def fake_sleep(seconds: float) -> None:
        clock["t"] += seconds

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    fake = FakeGemini(seed)

    class Models:
        def __init__(self, key: str) -> None:
            self.key = key

        async def generate_content(self, *, model: str, contents: list, config: dict):
            clock["t"] += 1.0
            kind = fake.reply(model, self.key)
            if kind == "ok":
                return SimpleNamespace(text="ok", usage_metadata=None)
            raise RuntimeError({"daily": f"429 RESOURCE_EXHAUSTED. {DAILY}",
                                "minute": f"429 RESOURCE_EXHAUSTED. {MINUTE}",
                                "busy": "503 UNAVAILABLE. The model is overloaded.",
                                "unknown": "404 NOT_FOUND. models/m6 is not found"}[kind])

    client = mod.GeminiClient(
        model=",".join(MODELS), api_keys=KEYS, cache_dir=tmp_path, rpm=60000,
        max_key_wait_s=300.0, overload_retries=5,
        sdk_factory=lambda k: SimpleNamespace(aio=SimpleNamespace(models=Models(k))),
    )
    client.pool.clock = lambda: clock["t"]
    ok = failed = 0
    per_call: list[int] = []
    for i in range(fake.capacity * 2 + 20):
        before = fake.requests
        try:
            await client.text_call("s", f"document {i}")  # distinct: no cached answer
            ok += 1
        except LLMUnreachable:
            failed += 1
        per_call.append(fake.requests - before)
    assert ok == fake.capacity, "a model or key was crossed off while it still had quota"
    assert max(per_call) <= 4 * len(MODELS) * len(KEYS), "a call kept retrying"
    assert max(per_call[-10:]) <= 6, "with everything spent, a call should fail at once"


# ------------------------------------------------------------------ question writer (labels, utility)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_question_writer_uses_every_quota_and_never_loops(seed, monkeypatch) -> None:
    from eval_suite import e2e

    fake = FakeGemini(seed)

    def fake_call(prompt: str, model: str, key: str) -> str:
        kind = fake.reply(model, key)
        if kind == "ok":
            return '{"ok": true}'
        code, body = {"daily": (429, DAILY), "minute": (429, MINUTE), "busy": (503, "overloaded"),
                      "unknown": (404, "not found")}[kind]
        raise urllib.error.HTTPError("u", code, body, {}, io.BytesIO(body.encode()))

    monkeypatch.setattr(e2e, "_gemini", fake_call)
    monkeypatch.setattr(e2e.time, "sleep", lambda s: None)
    monkeypatch.setenv("GEMINI_API_KEYS", ",".join(KEYS))
    monkeypatch.setenv("SANITISE_AGENTS_MODEL", ",".join(MODELS))
    monkeypatch.delenv("EVAL_LABEL_MODEL", raising=False)
    monkeypatch.delenv("EVAL_GEMINI_STOP_AT", raising=False)
    g = e2e.Gemini()
    g.models = list(MODELS)  # exactly the fake's models (no built-in defaults)
    ok = 0
    per_call: list[int] = []
    for i in range(fake.capacity + 30):
        before = fake.requests
        if g.ask("p", f"q{i}") is not None:
            ok += 1
        per_call.append(fake.requests - before)
    # it does not track daily limits per key, so it may miss a few: it must still get most
    assert ok >= 0.9 * fake.capacity, f"{ok} of {fake.capacity}: models left untried"
    assert max(per_call) <= max(24, 2 * len(MODELS) * len(KEYS)) + 1


def test_question_writer_stops_at_its_time_budget(monkeypatch) -> None:
    import time

    from eval_suite import e2e

    calls: list[str] = []
    monkeypatch.setattr(e2e, "_gemini", lambda p, m, k: calls.append(m) or '{"ok": 1}')
    monkeypatch.setenv("GEMINI_API_KEYS", "k1")
    monkeypatch.setenv("EVAL_GEMINI_STOP_AT", str(time.time() - 1))
    g = e2e.Gemini()
    assert g.ask("p", "q") is None and g.ask("p", "q2") is None
    assert calls == []  # the budget is spent: no request at all, the step finishes at once


# ------------------------------------------------------------------ build: one bad document


@pytest.mark.parametrize("pkg", ["sanitise", "sanitise_legal", "sanitise_finance"])
async def test_an_unexpected_error_fails_one_document_not_the_build(pkg) -> None:
    """An error of any type (a bug, odd JSON) used to end the build process: every pass then
    stopped at the same document and nothing after it was ever built."""
    import importlib

    build = importlib.import_module(f"{pkg}.build")
    Document = importlib.import_module(f"{pkg}.loader").Document

    class Store:
        def __init__(self) -> None:
            self.put_ids: list[str] = []

        def ids(self) -> list[str]:
            return []

        def put(self, doc_id, text, record) -> None:
            self.put_ids.append(doc_id)

    async def sanitiser(doc_id, text, llm):
        if doc_id == "bad":
            raise KeyError("facts")
        return SimpleNamespace(residual_leaks=[], model_dump_json=lambda **kw: "{}")

    docs = [Document(doc_id=d, text="t", source="s", sha256="0" * 64) for d in ("one", "bad", "two")]
    store = Store()
    report = await build.build_store(docs, store, None, sanitiser=sanitiser)
    assert store.put_ids == ["one", "two"]
    assert "KeyError" in report.failed["bad"]


def test_question_writer_moves_on_from_a_model_that_times_out(monkeypatch) -> None:
    from eval_suite import e2e

    asked: list[str] = []

    def fake(prompt: str, model: str, key: str) -> str:
        asked.append(model)
        if model == "slow":
            raise TimeoutError("The read operation timed out")
        return '{"ok": 1}'

    monkeypatch.setattr(e2e, "_gemini", fake)
    monkeypatch.setattr(e2e.time, "sleep", lambda s: None)
    monkeypatch.setenv("GEMINI_API_KEYS", "k1,k2")
    monkeypatch.delenv("EVAL_GEMINI_STOP_AT", raising=False)
    g = e2e.Gemini()
    g.models = ["slow", "fast"]
    assert g.ask("p", "q") == {"ok": 1}
    assert asked == ["slow", "fast"]  # one timeout, then the next model: not the same one again
