"""The retry / repair loop, exercised through MockLLM (same code path as Ollama)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sanitise.errors import LLMTimeout, SchemaValidationFailed
from sanitise.llm import LLMClient, MockLLM, track_calls
from sanitise.schemas import QuasiList

BAD_ENUM = '{"quasi_identifiers": [{"text": "x", "category": "DIAGNOSIS", "rationale": "r"}]}'
GOOD = {"quasi_identifiers": [{"text": "x", "category": "QUASI_COMBO", "rationale": "r"}]}


def test_satisfies_protocol() -> None:
    assert isinstance(MockLLM(), LLMClient)


async def test_default_reply_is_minimal_valid_instance() -> None:
    result = await MockLLM().json_call("sys", "user", QuasiList)
    assert result == QuasiList(quasi_identifiers=[])


async def test_repair_loop_recovers_and_escalates_temperature() -> None:
    llm = MockLLM(["not json at all", BAD_ENUM, GOOD], temperature=0.0, temperature_step=0.2)
    with track_calls() as stats:
        result = await llm.json_call("sys", "user", QuasiList, retries=3)
    assert result.quasi_identifiers[0].text == "x"
    assert [c.temperature for c in llm.calls] == pytest.approx([0.0, 0.2, 0.4])
    assert (stats.calls, stats.attempts, stats.retries, stats.failures) == (1, 3, 2, 0)

    first, second, third = llm.calls
    assert not first.is_repair
    # The repair turn replays the bad output and quotes the validation error.
    assert second.messages[2] == {"role": "assistant", "content": "not json at all"}
    assert "not valid JSON" in second.messages[3]["content"]
    assert "quasi_identifiers[0].category" in third.messages[3]["content"]
    assert "DIAGNOSIS" in third.messages[3]["content"]
    # Only the latest failure is kept, so the context does not grow.
    assert len(third.messages) == 4


async def test_gives_up_after_retries() -> None:
    llm = MockLLM([BAD_ENUM] * 5)
    with track_calls() as stats, pytest.raises(SchemaValidationFailed) as info:
        await llm.json_call("sys", "user", QuasiList, retries=2)
    assert info.value.attempts == 3 and len(llm.calls) == 3
    assert info.value.raw_output_sha256 is not None
    assert "DIAGNOSIS" in info.value.detail
    assert (stats.attempts, stats.retries, stats.failures) == (3, 2, 1)


async def test_temperature_is_capped() -> None:
    llm = MockLLM([BAD_ENUM] * 4, temperature=0.5, temperature_step=0.4, temperature_max=1.0)
    with pytest.raises(SchemaValidationFailed):
        await llm.json_call("s", "u", QuasiList, retries=3)
    assert [c.temperature for c in llm.calls] == pytest.approx([0.5, 0.9, 1.0, 1.0])


async def test_default_retry_budget_comes_from_client() -> None:
    llm = MockLLM([BAD_ENUM] * 5, retries=1)
    with pytest.raises(SchemaValidationFailed):
        await llm.json_call("s", "u", QuasiList)
    assert len(llm.calls) == 2

    strict = MockLLM([BAD_ENUM] * 5, retries=3).with_options(retries=0)
    with pytest.raises(SchemaValidationFailed):
        await strict.json_call("s", "u", QuasiList)
    assert len(strict.calls) == 1


async def test_scripted_exception_propagates() -> None:
    llm = MockLLM([LLMTimeout("too slow")])
    with pytest.raises(LLMTimeout):
        await llm.json_call("s", "u", QuasiList)


async def test_per_schema_queue_wins_over_shared_queue() -> None:
    llm = MockLLM(['{"quasi_identifiers": []}'], by_schema={"QuasiList": [GOOD]})
    assert len((await llm.json_call("s", "u", QuasiList)).quasi_identifiers) == 1
    assert len((await llm.json_call("s", "u", QuasiList)).quasi_identifiers) == 0
    assert llm.pending == 0


async def test_from_file(tmp_path: Path) -> None:
    path = tmp_path / "script.json"
    path.write_text(json.dumps({"QuasiList": [GOOD]}), encoding="utf-8")
    llm = MockLLM.from_file(path)
    assert (await llm.json_call("s", "u", QuasiList)).quasi_identifiers[
        0
    ].category.value == "QUASI_COMBO"


async def test_handler_sees_prompts() -> None:
    seen = []
    llm = MockLLM(
        handler=lambda call: seen.append((call.system, call.user, call.schema_name)) or GOOD
    )
    await llm.json_call("SYSTEM", "USER", QuasiList)
    assert seen == [("SYSTEM", "USER", "QuasiList")]


async def test_custom_repair_builder_is_used() -> None:
    llm = MockLLM([BAD_ENUM, GOOD], repair_builder=lambda error, raw, schema: f"FIX: {error}")
    await llm.json_call("s", "u", QuasiList)
    assert llm.calls[1].messages[3]["content"].startswith("FIX: quasi_identifiers[0].category")
