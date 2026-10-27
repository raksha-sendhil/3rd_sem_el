"""Attack runner.  OWNER: Person C (Platform and Integration).

Plays one attack script (data/attacks/*.json, written by Person D) against both pipelines,
one question per round, in a single session, and scores every round with Person D's
``score_round`` against the person table.

For ``interception`` the queries are ordinary users' questions; the attacker never prompts and
only reads the traffic to the model, so each round is scored on the payload instead of the
answer. ``iter_attack`` yields rounds as they finish (the dashboard streams them).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from sanitise.errors import AttackNotFound
from sanitise.evaluation.attacks import score_round
from sanitise.llm.base import LLMClient
from sanitise.rag.answer import RagContext, answer
from sanitise.schemas import AttackDefinition, AttackRound, AttackRun, Person, RagAnswer


def load_attacks(directory: str | Path) -> dict[str, AttackDefinition]:
    attacks: dict[str, AttackDefinition] = {}
    for path in sorted(Path(directory).glob("*.json")):
        attack = AttackDefinition.model_validate_json(path.read_text(encoding="utf-8"))
        attacks[attack.attack_id] = attack
    return attacks


def load_people(path: str | Path) -> list[Person]:
    path = Path(path)
    if not path.is_file():
        return []
    return [Person.model_validate(p) for p in json.loads(path.read_text(encoding="utf-8"))]


def get_attack(attacks: dict[str, AttackDefinition], attack_id: str) -> AttackDefinition:
    try:
        return attacks[attack_id]
    except KeyError:
        raise AttackNotFound(f"no attack {attack_id!r}", stage="attack") from None


def _observed(attack: AttackDefinition, answers: list[RagAnswer]) -> list[RagAnswer]:
    """What the attacker sees: the answers, or for interception the payloads on the wire."""
    if attack.kind != "interception":
        return answers
    return [a.model_copy(update={"answer": a.payload}) for a in answers]


async def iter_attack(
    ctx: RagContext,
    attack: AttackDefinition,
    people: list[Person],
    llm: LLMClient,
    session_id: str,
) -> AsyncIterator[AttackRound]:
    ctx.sessions.ensure(session_id)
    seen: dict[str, list[RagAnswer]] = {"baseline": [], "protected": []}
    for n, query in enumerate(attack.queries, start=1):
        base = await answer(ctx, query, "baseline", session_id, llm)
        prot = await answer(ctx, query, "protected", session_id, llm)
        seen["baseline"].append(base)
        seen["protected"].append(prot)
        round_ = AttackRound(
            round=n,
            query=query,
            baseline=base,
            protected=prot,
            baseline_leakage=score_round(
                people, attack.target_person_id, _observed(attack, seen["baseline"])
            ),
            protected_leakage=score_round(
                people, attack.target_person_id, _observed(attack, seen["protected"])
            ),
        )
        ctx.sessions.append(session_id, round_)
        yield round_


def summarise(attack: AttackDefinition, session_id: str, rounds: list[AttackRound]) -> AttackRun:
    def first(side: str) -> int | None:
        for r in rounds:
            leak = r.baseline_leakage if side == "baseline" else r.protected_leakage
            if leak is not None and leak.identified:
                return r.round
        return None

    return AttackRun(
        attack_id=attack.attack_id,
        session_id=session_id,
        rounds=rounds,
        baseline_identified_at=first("baseline"),
        protected_identified_at=first("protected"),
    )


async def run_attack(
    ctx: RagContext,
    attack: AttackDefinition,
    people: list[Person],
    llm: LLMClient,
    session_id: str | None = None,
) -> AttackRun:
    session_id = session_id or ctx.sessions.create()
    rounds = [r async for r in iter_attack(ctx, attack, people, llm, session_id)]
    return summarise(attack, session_id, rounds)
