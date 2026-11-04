"""Online path: store, retrieval, answers for both pipelines, sessions, attacks, scorers."""

from __future__ import annotations

import pytest

from sanitise.attacks.runner import _observed, load_attacks, load_people, run_attack
from sanitise.errors import DocNotFound, IndexNotBuilt, SessionNotFound
from sanitise.evaluation.attacks import score_round, target_info
from sanitise.evaluation.facts import score_facts
from sanitise.evaluation.qa import score_qa
from sanitise.evaluation.rewrite import score_rewrite
from sanitise.mock_mode import SAMPLE_DIR, build_mock_context, mock_llm, mock_sanitise
from sanitise.rag.answer import RagContext, answer, compare
from sanitise.rag.retrieve import BM25Retriever
from sanitise.rag.store import MemoryDocumentStore, SqliteDocumentStore
from sanitise.schemas import (
    ExtractorOutput,
    FactTuple,
    GoldRoute,
    GoldRouting,
    Importance,
    QAItem,
    RagAnswer,
)
from tests.conftest import SAMPLE_TEXT

DOCS = {
    "a": "The lighthouse keeper in Port Isaac has a rash on both hands.",
    "b": "A teacher in Pune asks about migraine and sleep.",
    "c": "My son plays kabaddi and tore a knee ligament last season.",
}


def test_bm25_ranks_the_matching_document_first() -> None:
    hits = BM25Retriever(DOCS).search("kabaddi ligament", k=2)
    assert hits[0].doc_id == "c" and hits[0].sparse_rank == 1
    assert BM25Retriever(DOCS).search("zzz", k=2) == []


async def test_stores_round_trip(fake_rules: None, tmp_path) -> None:
    record = await mock_sanitise("d", SAMPLE_TEXT, mock_llm())
    for store in (MemoryDocumentStore(), SqliteDocumentStore(tmp_path / "s.sqlite")):
        store.put("d", SAMPLE_TEXT, record)
        assert store.original("d") == SAMPLE_TEXT
        assert store.sanitised("d") == record and store.ids() == ["d"]
        with pytest.raises(DocNotFound):
            store.original("missing")


@pytest.fixture
async def ctx(fake_rules: None) -> RagContext:
    return await build_mock_context(mock_llm())


async def test_baseline_sees_originals_protected_sees_sanitised(ctx: RagContext) -> None:
    llm = mock_llm()
    session = ctx.sessions.create()
    result = await compare(ctx, "lighthouse keeper rash", session, llm)
    assert result.baseline.doc_ids == result.protected.doc_ids  # same retrieval
    doc = result.baseline.doc_ids[0]
    assert ctx.store.original(doc) in result.baseline.payload
    assert ctx.store.sanitised(doc).rewriter.safe_text in result.protected.payload
    assert result.protected.monitor.verdict == "allowed"
    assert len(ctx.sessions.get(session).rounds) == 1


async def test_empty_store_and_unknown_session() -> None:
    empty = RagContext(store=MemoryDocumentStore(), retriever=BM25Retriever({}))
    with pytest.raises(IndexNotBuilt):
        await answer(empty, "q", "baseline", "s", mock_llm())
    with pytest.raises(SessionNotFound):
        empty.sessions.get("nope")


async def test_run_sample_attack(ctx: RagContext) -> None:
    attacks = load_attacks(SAMPLE_DIR / "attacks")
    assert set(attacks) == {"context_dump_s01", "interception_s04", "stepwise_s02"}
    run = await run_attack(
        ctx, attacks["stepwise_s02"], load_people(SAMPLE_DIR / "people.json"), mock_llm()
    )
    assert len(run.rounds) == 4
    assert all(r.baseline_leakage and r.protected_leakage for r in run.rounds)
    assert len(ctx.sessions.get(run.session_id).rounds) == 4


def _answer(pipeline: str, query: str, text: str) -> RagAnswer:
    return RagAnswer(
        pipeline=pipeline, query=query, answer=text, doc_ids=[], payload="", latency_ms=0
    )


def test_score_round_counts_disclosure_and_pool() -> None:
    people = load_people(SAMPLE_DIR / "people.json")
    nothing = score_round(people, "s01", [_answer("baseline", "q", "No records match.")])
    assert nothing.disclosed == [] and not nothing.identified
    named = score_round(people, "s01", [_answer("baseline", "q", "That is Priya Raman.")])
    assert "name" in named.disclosed and named.identified
    town = score_round(people, "s01", [_answer("protected", "q", "Someone in Coonoor.")])
    assert town.disclosed == ["town"] and town.candidate_pool == 3
    assert town.recovery_rate == pytest.approx(1 / 6)  # name, age, town, job, phone, vertigo


def test_interception_is_scored_on_the_payload_not_the_answer() -> None:
    people = load_people(SAMPLE_DIR / "people.json")
    attack = load_attacks(SAMPLE_DIR / "attacks")["interception_s04"]
    sent = _answer("baseline", "q", "The records do not say.").model_copy(
        update={"payload": "CONTEXT: Cyril D'Souza, 71, Fort Kochi, ischemic stroke"}
    )
    seen = score_round(people, "s04", _observed(attack, [sent]))
    assert seen.identified and {"name", "town"} <= set(seen.disclosed)
    dump = load_attacks(SAMPLE_DIR / "attacks")["context_dump_s01"]
    assert score_round(people, "s04", _observed(dump, [sent])).disclosed == []


def test_target_info_counts_true_values_over_everyone() -> None:
    people = load_people(SAMPLE_DIR / "people.json")
    said = [_answer("baseline", "q", "Priya Raman from Coonoor; Arjun Nair is 26 years old.")]
    assert target_info(people, said) >= 4
    assert target_info(people, [_answer("protected", "q", "[PERSON_1] lives in a town.")]) == 0


async def test_answers_use_the_spec_generation_settings(ctx: RagContext) -> None:
    llm = mock_llm()
    await answer(ctx, "vertigo", "protected", ctx.sessions.create(), llm)
    assert llm.calls[-1].temperature == 0.4


def test_placeholder_scorers() -> None:
    gold = [
        ExtractorOutput(
            doc_id="d",
            facts=[
                FactTuple(
                    subject="writer",
                    relation="has_condition",
                    value="vertigo",
                    importance=Importance.HIGH,
                )
            ],
        )
    ]
    assert score_facts(gold, gold).coverage == 1.0
    assert score_facts([ExtractorOutput(doc_id="d", facts=[])], gold).coverage == 0.0
    qa_set = [QAItem(qa_id="q1", question="What does she have?", expected=["vertigo"])]
    m = score_qa(
        [
            _answer("baseline", "What does she have?", "Vertigo."),
            _answer("protected", "What does she have?", "A condition."),
        ],
        qa_set,
    )
    assert (m.baseline_accuracy, m.protected_accuracy) == (1.0, 0.0)
    from sanitise.schemas import Action, PIICategory, RewriterOutput

    routing = [
        GoldRouting(
            doc_id="d",
            routing=[GoldRoute(text="Priya", category=PIICategory.NAME, action=Action.PLACEHOLDER)],
        )
    ]
    pred = [
        RewriterOutput(doc_id="d", safe_text="Priya has vertigo.", decisions=[], placeholders={})
    ]
    r = score_rewrite(pred, routing, gold)
    assert (r.leak_rate, r.utility, r.routing_accuracy) == (1.0, 1.0, 0.0)


def test_the_real_attack_folder_loads_and_skips_the_target_info_query_list():
    attacks = load_attacks(SAMPLE_DIR.parent / "attacks")
    assert len(attacks) == 9
    assert {a.kind for a in attacks.values()} == {"context_dump", "stepwise", "interception"}
