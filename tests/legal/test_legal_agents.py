"""The legal domain in the agents (C's prompts, A's routing), clients, replay and the API."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from sanitise.agents.detector import quasi
from sanitise.agents.extractor import prompt as extract_prompt
from sanitise.agents.rewriter import prompt as rewrite_prompt
from sanitise.agents.rewriter.routing import route
from sanitise.domain import current_domain, use_domain
from sanitise.legal.export import RecordedRetriever, _key, replay_llm
from sanitise.llm.gemini import GeminiClient
from sanitise.llm.mock import MockLLM
from sanitise.rag import answer_prompt
from sanitise.rag.retrieve import BM25Retriever
from sanitise.schemas import PIICategory as C
from sanitise.schemas import SensitiveSpan

REPO = Path(__file__).resolve().parents[2]


def span(text: str, category: C, start: int) -> SensitiveSpan:
    return SensitiveSpan(
        text=text, category=category, start=start, end=start + len(text), method="rule"
    )


def test_domain_switch() -> None:
    assert current_domain() == "medical"  # tests/conftest.py pins the older suite
    with use_domain("legal"):
        assert current_domain() == "legal"
        assert answer_prompt.build_answer_prompt("q", [])[0] == answer_prompt.LEGAL_SYSTEM_PROMPT
    assert answer_prompt.build_answer_prompt("q", [])[0] == answer_prompt.SYSTEM_PROMPT


async def test_legal_llm_pass_grounds_every_quoted_span() -> None:
    text = "Our client Kavya Iyer, a dentist from Kochi, sued Lotus Realty over a land dispute."
    reply = {
        "spans": [
            {"text": "Kavya Iyer", "category": "NAME"},
            {"text": "dentist", "category": "OCCUPATION"},
            {"text": "Lotus Realty", "category": "ORGANIZATION"},
            {"text": "land dispute", "category": "LEGAL_ISSUE"},
            {"text": "Atlantis", "category": "LOCATION"},  # not in the text: dropped
            {"text": "occupation (dentist) + city (Kochi)", "category": "QUASI_COMBO"},
        ]
    }
    llm = MockLLM(by_schema={"SpanList": [reply]})
    with use_domain("legal"):
        spans = await quasi.infer_quasi(text, [], llm)
    assert llm.calls[-1].system == quasi.LEGAL_SYSTEM_PROMPT
    got = {(s.text, s.category) for s in spans}
    assert ("Lotus Realty", C.ORGANIZATION) in got and ("land dispute", C.LEGAL_ISSUE) in got
    assert all(text[s.start : s.end] == s.text for s in spans if s.start is not None)
    assert not any(s.text == "Atlantis" for s in spans)


def test_legal_prompt_examples_parse() -> None:
    block = re.search(r"OUTPUT:\n(\{.*?\]\})", quasi.LEGAL_SYSTEM_PROMPT, re.S).group(1)
    assert len(quasi.LegalSpanList.model_validate_json(block).spans) == 10
    for block in re.findall(r"OUTPUT:\n(\{.*?\]\})", extract_prompt.LEGAL_SYSTEM_PROMPT, re.S):
        extract_prompt.LLMFactList.model_validate_json(block)


@pytest.mark.parametrize(
    "prompt",
    [
        quasi.LEGAL_SYSTEM_PROMPT,
        extract_prompt.LEGAL_SYSTEM_PROMPT,
        rewrite_prompt.LEGAL_SYSTEM_PROMPT,
        answer_prompt.LEGAL_SYSTEM_PROMPT,
    ],
)
def test_no_legal_evaluation_text_in_the_prompts(prompt: str) -> None:
    """The legal metrics run on data/legal; its names, firms and numbers must not be here."""
    lowered = prompt.lower()
    leaks = set()
    for path in (REPO / "data" / "legal" / "gold").glob("*.json"):
        for s in json.loads(path.read_text(encoding="utf-8"))["spans"]:
            identifying = s["category"] in {
                "NAME",
                "ORGANIZATION",
                "CASE_NUMBER",
                "PHONE",
                "EMAIL",
                "ID_NUMBER",
                "ADDRESS",
                "MATTER_CODENAME",
            }
            if identifying and re.search(rf"\b{re.escape(s['text'].lower())}\b", lowered):
                leaks.add(s["text"])
    assert not leaks, sorted(leaks)


def test_implicit_combination_climbs_two_steps_in_the_legal_domain() -> None:
    spans = [
        span("52-year-old", C.AGE, 0),
        span("cardiologist", C.OCCUPATION, 20),
        span("Pune", C.LOCATION, 40),
        span("cheque bounce", C.LEGAL_ISSUE, 60),
    ]
    with use_domain("legal"):
        legal = [d.replacement for d in route(spans, [])]
    assert legal == ["middle-aged", "a professional", "a city in western India", None]
    assert [d.replacement for d in route(spans, [])] == [None, None, None, None]  # medical
    with use_domain("legal"):
        two = [d.replacement for d in route(spans[1:3], [])]
    assert two == [None, None]  # fewer than three kinds: one step, filled in later


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate_content(self, *, model: str, contents: list, config: dict):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text='{"facts": []}' if "response_mime_type" in config else "ok")


async def test_gemini_client_caches_and_maps_messages(tmp_path: Path) -> None:
    models = _FakeModels()
    sdk = SimpleNamespace(aio=SimpleNamespace(models=models))
    client = GeminiClient(cache_dir=tmp_path, rpm=6000, sdk_client=sdk)
    assert await client.text_call("be brief", "hi", temperature=0.4) == "ok"
    assert await client.text_call("be brief", "hi", temperature=0.4) == "ok"
    assert len(models.calls) == 1 and client.cache_hits == 1
    call = models.calls[0]
    assert call["model"] == "gemini-3.5-flash"
    assert call["config"]["system_instruction"] == "be brief"
    assert call["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    out = await client.json_call("s", "u", extract_prompt.LLMFactList)
    assert out.facts == [] and models.calls[-1]["config"]["response_mime_type"] == (
        "application/json"
    )


async def test_gemini_without_a_key_is_unreachable(monkeypatch, tmp_path: Path) -> None:
    from sanitise.errors import LLMUnreachable
    from sanitise.llm.gemini import KEY_NAMES

    for name in (*KEY_NAMES, "GEMINI_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMUnreachable):
        await GeminiClient(cache_dir=tmp_path).text_call("s", "u")


class _KeyedModels:
    """A fake SDK per key: the keys in ``limited`` answer with a quota error."""

    def __init__(self, key: str, limited: dict[str, str], log: list[str]) -> None:
        self.key, self.limited, self.log = key, limited, log

    async def generate_content(self, *, model: str, contents: list, config: dict):
        self.log.append(self.key)
        if self.key in self.limited:
            raise RuntimeError(self.limited[self.key])
        usage = SimpleNamespace(total_token_count=700)
        return SimpleNamespace(text=f"answer from {self.key}", usage_metadata=usage)


def _rotating(tmp_path: Path, keys: list[str], limited: dict[str, str], **kw) -> tuple:
    log: list[str] = []
    client = GeminiClient(
        api_keys=keys,
        cache_dir=tmp_path,
        rpm=6000,
        sdk_factory=lambda k: SimpleNamespace(
            aio=SimpleNamespace(models=_KeyedModels(k, limited, log))
        ),
        **kw,
    )
    return client, log


async def test_a_rate_limited_key_hands_over_to_the_next(tmp_path: Path) -> None:
    client, log = _rotating(
        tmp_path, ["k1", "k2"], {"k1": "429 RESOURCE_EXHAUSTED: requests per minute"}
    )
    assert await client.text_call("s", "u1") == "answer from k2"
    assert await client.text_call("s", "u2") == "answer from k2"  # k1 is resting
    assert log == ["k1", "k2", "k2"]
    usage = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    assert "k2" not in json.dumps(usage)  # fingerprints only, never the key


async def test_daily_quota_and_token_budget_rest_a_key_for_the_day(tmp_path: Path) -> None:
    from sanitise.errors import LLMUnreachable

    client, _ = _rotating(
        tmp_path, ["k1", "k2"], {"k1": "429 quota exceeded: GenerateRequestsPerDay"}
    )
    assert await client.text_call("s", "u1") == "answer from k2"
    assert client.pool._keys[0].rest_until - client.pool.clock() > 3600  # a day, not a minute

    budget, log = _rotating(tmp_path / "b", ["k1", "k2"], {}, daily_token_budget=1000)
    for i in range(4):  # 700 tokens per call: each key takes two calls, then is over budget
        await budget.text_call("s", f"u{i}")
    assert log == ["k1", "k2", "k1", "k2"]
    with pytest.raises(LLMUnreachable, match="over their daily budget"):
        await budget.text_call("s", "u5")


async def test_all_keys_limited_fails_clearly(tmp_path: Path) -> None:
    from sanitise.errors import LLMUnreachable

    limited = {"k1": "429 rate limit", "k2": "429 rate limit"}
    client, log = _rotating(tmp_path, ["k1", "k2"], limited)
    with pytest.raises(LLMUnreachable, match="all 2 Gemini keys"):
        await client.text_call("s", "u")
    assert log == ["k1", "k2"]


async def test_a_short_rest_is_waited_out_instead_of_failing(tmp_path: Path, monkeypatch) -> None:
    client, log = _rotating(tmp_path, ["k1"], {"k1": "429 rate limit"}, max_key_wait_s=60)
    now = [1000.0]
    client.pool.clock = lambda: now[0]
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds
        client.pool._keys[0].sdk.aio.models.limited = {}  # the limit has passed

    monkeypatch.setattr("sanitise.llm.gemini.asyncio.sleep", sleep)
    assert await client.text_call("s", "u") == "answer from k1"
    assert log == ["k1", "k1"] and slept.count(60.0) == 1


async def test_a_daily_quota_is_not_waited_for(tmp_path: Path) -> None:
    from sanitise.errors import LLMUnreachable

    limited = {"k1": "429 quota exceeded: GenerateRequestsPerDay"}
    client, log = _rotating(tmp_path, ["k1"], limited, max_key_wait_s=300)
    with pytest.raises(LLMUnreachable, match="all 1 Gemini keys"):
        await client.text_call("s", "u")
    assert log == ["k1"]


async def test_a_limit_that_never_lifts_fails_in_the_end(tmp_path: Path, monkeypatch) -> None:
    from sanitise.errors import LLMUnreachable

    client, log = _rotating(
        tmp_path, ["k1"], {"k1": "429 rate limit"}, max_key_wait_s=60, overload_retries=2
    )
    now = [1000.0]
    client.pool.clock = lambda: now[0]

    async def sleep(seconds: float) -> None:
        now[0] += seconds

    monkeypatch.setattr("sanitise.llm.gemini.asyncio.sleep", sleep)
    with pytest.raises(LLMUnreachable, match="all 1 Gemini keys"):
        await client.text_call("s", "u")
    assert len(log) == 3  # the first try and one after each of the two waits


async def test_an_empty_reply_is_not_cached(tmp_path: Path) -> None:
    replies = ["", "ok"]

    async def generate_content(*, model: str, contents: list, config: dict):
        return SimpleNamespace(text=replies.pop(0))

    sdk = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    )
    client = GeminiClient(cache_dir=tmp_path, rpm=6000, sdk_client=sdk)
    assert await client.text_call("s", "u") == ""
    assert await client.text_call("s", "u") == "ok"


def test_the_factory_builds_gemini_for_every_agent(tmp_path: Path) -> None:
    from sanitise.config import Settings
    from sanitise.domain import settings_for
    from sanitise.llm.factory import build_agent1_llm, build_answer_llm, build_llm_client

    base = Settings(llm_backend="ollama", gemini_api_keys="k", gemini_cache_dir=str(tmp_path))
    s = settings_for(base, "legal")
    agents = build_llm_client(s)
    assert isinstance(agents, GeminiClient) and agents.model == "gemini-3.5-flash"
    assert build_agent1_llm(s, agents) is agents
    assert build_answer_llm(s, agents) is agents  # one key pool and one rate limit for all


def test_a_global_mock_backend_beats_the_profile() -> None:
    from sanitise.config import Settings
    from sanitise.domain import settings_for

    assert settings_for(Settings(llm_backend="mock"), "legal").llm_backend == "mock"


HIGH_DEMAND = "503 UNAVAILABLE. {'error': {'message': 'experiencing high demand'}}"


def _overloaded_sdk(failures: int, calls: list[int]) -> SimpleNamespace:
    async def generate_content(*, model: str, contents: list, config: dict):
        calls.append(1)
        if len(calls) <= failures:
            raise RuntimeError(HIGH_DEMAND)
        return SimpleNamespace(text="ok")

    return SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    )


async def test_an_overloaded_model_is_waited_for_on_the_same_call(tmp_path: Path) -> None:
    calls: list[int] = []
    client = GeminiClient(
        cache_dir=tmp_path, rpm=6000, overload_wait_s=0, sdk_client=_overloaded_sdk(3, calls)
    )
    assert await client.text_call("s", "u") == "ok"
    assert len(calls) == 4
    assert client.pool._keys[0].rest_until == 0.0  # the key did nothing wrong


async def test_an_overload_that_never_ends_fails_and_caches_nothing(tmp_path: Path) -> None:
    from sanitise.errors import LLMUnreachable

    calls: list[int] = []
    client = GeminiClient(
        cache_dir=tmp_path,
        rpm=6000,
        overload_retries=2,
        overload_wait_s=0,
        sdk_client=_overloaded_sdk(99, calls),
    )
    with pytest.raises(LLMUnreachable, match="503"):
        await client.text_call("s", "u")
    assert len(calls) == 3
    with pytest.raises(LLMUnreachable, match="503"):  # not served from the cache
        await client.text_call("s", "u")
    assert len(calls) == 6


async def test_other_gemini_errors_are_not_retried(tmp_path: Path) -> None:
    from sanitise.errors import LLMUnreachable

    client, log = _rotating(tmp_path, ["k1"], {"k1": "400 INVALID_ARGUMENT: max 500 tokens"})
    with pytest.raises(LLMUnreachable, match="400"):
        await client.text_call("s", "u")
    assert log == ["k1"]


def test_keys_are_read_from_numbered_variables(monkeypatch) -> None:
    from sanitise.llm.gemini import gemini_api_keys

    monkeypatch.setenv("GEMINI_API_KEYS", "a, b")
    monkeypatch.setenv("GEMINI_API_KEY", "c")
    monkeypatch.setenv("GEMINI_API_KEY_3", "a")
    assert gemini_api_keys("x") == ["x", "a", "b", "c"]


def test_domains_pick_their_own_models_and_data() -> None:
    from sanitise.config import Settings
    from sanitise.domain import profiles, settings_for

    base = Settings(domain="medical", model="llama3:8b", llm_backend="ollama")
    assert {"medical", "legal"} <= set(profiles(base))
    legal = settings_for(base, "legal")
    assert (legal.llm_backend, legal.answer_backend, legal.retrieval_k) == ("gemini", "same", 4)
    assert legal.model == legal.gemini_model == "gemini-3.5-flash"
    assert legal.store_path == "data/legal/store.sqlite"
    medical = settings_for(base, "medical")
    assert (medical.model, medical.answer_backend, medical.retrieval_k) == ("llama3:8b", "same", 3)
    assert medical.llm_backend == "ollama"


def test_a_domain_added_in_the_file_is_served(tmp_path: Path, monkeypatch) -> None:
    from sanitise.config import Settings
    from sanitise.domain import profiles, settings_for

    file = tmp_path / "domains.json"
    file.write_text(
        json.dumps({"finance": {"label": "Finance", "prompts": "legal", "model": "gemma4:e4b"}}),
        encoding="utf-8",
    )
    s = Settings(domains_file=str(file))
    assert set(profiles(s)) == {"medical", "legal", "finance"}
    assert settings_for(s, "finance").model == "gemma4:e4b"
    with use_domain("finance"), pytest.raises(ValueError):
        settings_for(s, "nope")


def test_api_switches_domain_per_request(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from sanitise.api.app import create_app

    with TestClient(create_app()) as client:
        listed = client.get("/v1/domains").json()
        assert {d["name"] for d in listed["domains"]} >= {"medical", "legal"}
        legal = {a["kind"] for a in client.get("/v1/attacks?domain=legal").json()}
        medical = {a["kind"] for a in client.get("/v1/attacks?domain=medical").json()}
        header = client.get("/v1/attacks", headers={"X-Sanitise-Domain": "legal"}).json()
        bad = client.get("/v1/attacks?domain=astrology")
    assert legal == {"direct", "linkage", "stepwise"}
    assert medical == {"context_dump", "stepwise", "interception"}
    assert {a["kind"] for a in header} == legal
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "UNKNOWN_DOMAIN"


async def test_replay_returns_recorded_answers(tmp_path: Path) -> None:
    hits = [{"doc_id": "d2", "score": 1.0, "dense_rank": None, "sparse_rank": 1}]
    (tmp_path / "recorded_calls.json").write_text(
        json.dumps(
            {
                "llm": [{"system": "S", "user": "U", "text": "recorded"}],
                "retrieval": {"4\twho?": hits},
            }
        ),
        encoding="utf-8",
    )
    llm = replay_llm(tmp_path)
    assert await llm.text_call("S", "U") == "recorded"
    assert (await llm.text_call("S", "CONTEXT:\n[d]\nfirst. second.\nQUESTION: x")).startswith(
        "According to the records"
    )
    retriever = RecordedRetriever({"4\twho?": hits}, BM25Retriever({"d1": "alpha", "d2": "beta"}))
    assert [h.doc_id for h in retriever.search("who?", 4)] == ["d2"]
    assert [h.doc_id for h in retriever.search("alpha", 4)] == ["d1"]
    assert _key("a", "b") != _key("ab", "")


def test_api_lists_the_legal_attacks_in_legal_mock_mode(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from sanitise.api.app import create_app
    from sanitise.config import get_settings

    monkeypatch.setenv("SANITISE_DOMAIN", "legal")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        kinds = {a["kind"] for a in client.get("/v1/attacks").json()}
    assert kinds == {"direct", "linkage", "stepwise"}
