"""Broken model output in the legal domain: fall back, record it, keep the document."""

from __future__ import annotations

import json

import httpx
import pytest

from sanitise.agents.detector.quasi import LegalSpanList, LLMQuasiList
from sanitise.agents.extractor.prompt import LLMFactList
from sanitise.agents.pipeline import sanitise
from sanitise.domain import use_domain
from sanitise.errors import LLMUnreachable, SchemaValidationFailed
from sanitise.llm.ollama import OllamaClient

TEXT = (
    "My son has been experiencing left-sided abdominal pain for about ten weeks. "
    "He is 32 years old, 6 ft 3 inches tall, and weighs approximately 220 pounds. "
    "He works as a lighthouse keeper in Port Isaac. You can reach me on 07700 900123."
)
ADVICE = "send a statutory demand notice to the drawer"


class FakeLLM:
    """Scripted agents: ``broken`` lists the schemas that always fail."""

    model = "fake"

    def __init__(self, broken=(), rewrite="[PHONE_1] was advised to " + ADVICE, error=None):
        self.broken, self.rewrite, self.error = set(broken), rewrite, error

    async def json_call(self, system, user, schema, retries=None):
        if self.error:
            raise self.error
        if schema.__name__ in self.broken:
            raise SchemaValidationFailed(f"{schema.__name__}: broken", attempts=2)
        if schema is LegalSpanList:
            return LegalSpanList(spans=[])
        if schema is LLMQuasiList:
            return LLMQuasiList(quasi_identifiers=[])
        return LLMFactList(
            facts=[
                {"subject": "client", "relation": "advised", "value": ADVICE, "importance": "high"}
            ]
        )

    async def text_call(self, system, user, **options):
        return self.rewrite


@pytest.mark.usefixtures("fake_rules")
async def test_every_model_step_working_uses_no_fallback() -> None:
    with use_domain("legal"):
        record = await sanitise("d", TEXT, FakeLLM())
    assert record.meta.fallback == []
    assert ADVICE in record.rewriter.safe_text


@pytest.mark.usefixtures("fake_rules")
async def test_broken_agent1_falls_back_to_rules() -> None:
    with use_domain("legal"):
        record = await sanitise("d", TEXT, FakeLLM(broken={"LegalSpanList"}))
    assert record.meta.fallback == ["agent1_rules_only"]
    assert any(s.text == "07700 900123" for s in record.detector.spans)


@pytest.mark.usefixtures("fake_rules")
async def test_broken_agent2_rewrites_in_place_without_identifiers() -> None:
    with use_domain("legal"):
        record = await sanitise("d", TEXT, FakeLLM(broken={"LLMFactList"}))
    assert record.meta.fallback == ["agent2_rules_only", "agent3_in_place"]
    assert "07700 900123" not in record.rewriter.safe_text
    assert "lighthouse keeper" in record.rewriter.safe_text  # the rest of the text is kept
    assert record.residual_leaks == []


@pytest.mark.usefixtures("fake_rules")
@pytest.mark.parametrize("rewrite", ["", "The client was told something else entirely."])
async def test_a_rewrite_that_drops_the_advice_is_replaced(rewrite: str) -> None:
    with use_domain("legal"):
        record = await sanitise("d", TEXT, FakeLLM(rewrite=rewrite))
    assert record.meta.fallback == ["agent3_in_place"]
    assert "07700 900123" not in record.rewriter.safe_text


@pytest.mark.usefixtures("fake_rules")
async def test_no_facts_left_falls_back_rather_than_an_empty_record() -> None:
    class NoFacts(FakeLLM):
        async def json_call(self, system, user, schema, retries=None):
            if schema is LLMFactList:
                return LLMFactList(facts=[])
            return await super().json_call(system, user, schema, retries)

    with use_domain("legal"):
        record = await sanitise("d", TEXT, NoFacts())
    assert record.meta.fallback == ["agent3_in_place"]
    assert "lighthouse keeper" in record.rewriter.safe_text


@pytest.mark.usefixtures("fake_rules")
async def test_an_unreachable_model_still_fails_the_document() -> None:
    with use_domain("legal"), pytest.raises(LLMUnreachable):
        await sanitise("d", TEXT, FakeLLM(error=LLMUnreachable("down")))


@pytest.mark.usefixtures("fake_rules")
async def test_the_medical_domain_does_not_fall_back() -> None:
    with pytest.raises(SchemaValidationFailed):
        await sanitise("d", TEXT, FakeLLM(broken={"LLMFactList"}))


async def test_think_is_sent_only_when_set() -> None:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    for think in (False, None):
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://o")
        await OllamaClient(think=think, http=http).text_call("s", "u")
        await http.aclose()
    assert bodies[0]["think"] is False and "think" not in bodies[1]


def test_the_legal_profile_turns_thinking_off() -> None:
    from sanitise.config import Settings
    from sanitise.domain import settings_for

    legal = settings_for(Settings(domain="medical"), "legal")
    assert (legal.llm_think, legal.llm_retries, legal.llm_num_predict) == (False, 1, 2048)
    assert settings_for(Settings(domain="medical"), "medical").llm_think is None
