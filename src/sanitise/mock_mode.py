"""Mock mode: the whole Phase 3a API without a model.  OWNER: Person C.

With ``SANITISE_LLM_BACKEND=mock``, ``sanitise serve`` answers every endpoint in
docs/phase3_plan.md 6.4 from deterministic stand-ins:

* ``mock_sanitise``: Agent 1's rule pass (real) and LLM pass (MockLLM, so nothing), facts made
  directly from the detected spans, Person A's routing and placeholders, and
  ``apply_decisions`` instead of the reconstruction model. the LLM prompts are not used,
  so mock mode keeps working whatever state they are in.
* ``mock_llm``: a MockLLM whose free-text replies quote the first sentences of the first
  record in the prompt, so baseline answers show original details and protected answers show
  placeholders, which is the contrast the dashboard is about.
* the corpus is ``data/fixtures`` and the people and attacks come from ``data/sample``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from sanitise.agents.detector.pipeline import detect
from sanitise.agents.extractor import canonical, importance
from sanitise.agents.rewriter import leakcheck
from sanitise.agents.rewriter.pipeline import apply_decisions, plan_decisions, removed_decisions
from sanitise.llm.base import LLMClient
from sanitise.llm.mock import TEXT_CALL, MockCall, MockLLM, fabricate
from sanitise.loader import sha256_hex
from sanitise.rag.answer import RagContext
from sanitise.rag.retrieve import BM25Retriever
from sanitise.rag.store import MemoryDocumentStore
from sanitise.schemas import (
    DetectorOutput,
    ExtractorOutput,
    FactTuple,
    PIICategory,
    RewriterOutput,
    SanitisedRecord,
    SanitiseMeta,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "data" / "fixtures"
SAMPLE_DIR = REPO_ROOT / "data" / "sample"

C = PIICategory
_RELATION_FOR: dict[PIICategory, str] = {
    C.AGE: "age",
    C.DATE: "event_date",
    C.LOCATION: "lives_in",
    C.ADDRESS: "lives_in",
    C.OCCUPATION: "works_as",
    C.MEDICAL_CONDITION: "has_condition",
    C.SYMPTOM: "has_symptom",
    C.PHYSICAL: "height",
    C.NAME: "identifier",
    C.PHONE: "contact",
    C.EMAIL: "contact",
    C.ID_NUMBER: "identifier",
    C.RELATION: "relation_to",
}


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def mock_facts(detected: DetectorOutput) -> list[FactTuple]:
    """One fact per positional span, attached to the writer."""
    facts = []
    for span in detected.spans:
        relation = _RELATION_FOR.get(span.category)
        if relation is None or span.start is None:
            continue
        facts.append(
            FactTuple(
                subject="writer",
                relation=relation,
                value=span.text,
                importance=importance.default_importance(relation),
            )
        )
    return facts


async def mock_sanitise(doc_id: str, text: str, llm: LLMClient) -> SanitisedRecord:
    t0 = time.perf_counter()
    detected = await detect(doc_id, text, llm)
    detect_ms = _ms(t0)
    subs = canonical.subjects(text, detected.output.spans)
    extracted = ExtractorOutput(doc_id=doc_id, facts=mock_facts(detected.output))
    t = time.perf_counter()
    decisions, tokens = plan_decisions(detected.output, extracted)
    safe = apply_decisions(text, decisions)
    leaks = leakcheck.leak_check(removed_decisions(decisions), safe)
    rewriter = RewriterOutput(
        doc_id=doc_id, safe_text=safe, decisions=decisions, placeholders=tokens
    )
    return SanitisedRecord(
        doc_id=doc_id,
        original_sha256=sha256_hex(text),
        subjects=subs,
        detector=detected.output,
        extractor=extracted,
        rewriter=rewriter,
        residual_leaks=leaks,
        meta=SanitiseMeta(
            model="mock",
            detect_ms=detect_ms,
            extract_ms=0,
            rewrite_ms=_ms(t),
            total_ms=_ms(t0),
            retries=0,
            leak_retries=0,
        ),
    )


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _canned_answer(user_message: str) -> str:
    """Quote the first two sentences of the first record in the prompt."""
    for line in user_message.splitlines():
        line = line.strip()
        if (
            not line
            or line.endswith(":")
            or line.startswith("QUESTION")
            or (line.startswith("[") and line.endswith("]"))
        ):
            continue
        sentences = _SENTENCE.split(line)
        return "According to the records: " + " ".join(sentences[:2])
    return "The records do not contain an answer to this question."


def _handler(call: MockCall) -> Any:
    if call.schema_name == TEXT_CALL:
        return _canned_answer(call.user)
    return fabricate(call.json_schema)


def mock_llm() -> MockLLM:
    return MockLLM(handler=_handler, model="mock-llm")


def _read_fixtures(fixtures_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(fixtures_dir.glob("*.json"))]


async def build_mock_context(llm: LLMClient, fixtures_dir: Path = FIXTURES_DIR) -> RagContext:
    store = MemoryDocumentStore()
    originals: dict[str, str] = {}
    for doc in _read_fixtures(fixtures_dir):
        record = await mock_sanitise(doc["doc_id"], doc["text"], llm)
        store.put(doc["doc_id"], doc["text"], record)
        originals[doc["doc_id"]] = doc["text"]
    return RagContext(store=store, retriever=BM25Retriever(originals))
