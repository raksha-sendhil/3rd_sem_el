"""A strong model does Agent 1's whole detection; a small model keeps the narrow prompt."""

import asyncio

from sanitise.agents.detector import quasi
from sanitise.schemas import PIICategory, SensitiveSpan


class FakeLLM:
    def __init__(self, strong: bool, answer: dict):
        self.one_call_per_document = strong
        self.answer = answer
        self.calls: list[tuple[str, str, type]] = []

    async def json_call(self, system, user, schema):
        self.calls.append((system, user, schema))
        return schema.model_validate(self.answer)


TEXT = "From: Lena Ortiz\nSent: 3/11\nCall 555-0142. My son has a cough."
RULE = [SensitiveSpan(text="555-0142", category=PIICategory.PHONE, start=29, end=37, method="rule")]


def test_strong_model_gets_every_category_and_hints():
    llm = FakeLLM(
        True,
        {
            "details": [
                {"text": "Lena Ortiz", "category": "NAME", "rationale": "sender"},
                {"text": "3/11", "category": "DATE", "rationale": "sent"},
                {"text": "My son", "category": "RELATION", "rationale": "son"},
                {"text": "cough", "category": "SYMPTOM", "rationale": "son's symptom"},
            ]
        },
    )
    spans = asyncio.run(quasi.infer_quasi(TEXT, RULE, llm))
    system, user, schema = llm.calls[0]
    assert schema is quasi.FullDetection and "HINTS:\n- PHONE: 555-0142" in user
    assert "ANY person" in system
    got = {(s.text, s.category.value) for s in spans}
    assert got == {("Lena Ortiz", "NAME"), ("3/11", "DATE"), ("son", "RELATION"),
                   ("cough", "SYMPTOM")}  # fmt: skip
    assert all(TEXT[s.start : s.end] == s.text for s in spans)


def test_small_model_keeps_the_narrow_prompt():
    llm = FakeLLM(False, {"quasi_identifiers": []})
    asyncio.run(quasi.infer_quasi(TEXT, RULE, llm))
    assert llm.calls[0][2] is quasi.LLMQuasiList
